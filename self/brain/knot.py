import torch
from torch._C import dtype
from torch.functional import Tensor
import torch.nn as nn

#https://pytorch.org/docs/stable/jit_language_reference.html
from typing import Dict, List, Tuple
import math

from torch.nn.modules.linear import Identity

DEFAULT_FFT_SAMPLES = 128


@torch.jit.script
def irregularGauss(x: torch.Tensor, mean: torch.Tensor, lowStd: torch.Tensor, highStd: torch.Tensor) -> torch.Tensor:
  """Generates an piecewise Gaussian curve according to the provided parameters.

  Args:
      x (torch.Tensor): The sampling value for the curve with indefinite size.
      mean (torch.Tensor): The means that generate the peaks of the function which
        has a shape that is broadcastable upon x.
      lowStd (torch.Tensor): The standard deviation to use when the function is below
        the defined mean. The size must be broadcastable upon x.
      highStd (torch.Tensor): The standard deviation to use when the function is
        above the defined mean. The size must be broadcastable upon x.

  Returns:
      torch.Tensor: A sampled set of values with the same size as the input.
  """
  # Grab the correct side of the curve
  if x <= mean: std = lowStd
  else: std = highStd

  # Never hits 0 or inf., easy to take derivative
  std = torch.exp(std)

  # Calculate the gaussian curve
  top = torch.square(x - mean)
  bottom = torch.square(std)
  return torch.exp((-0.5) * (top / bottom))

class LinearGauss(nn.Module):
  """
  A linearly tuned irregular gaussian function to be used as an activation layer of sorts.
  """
  def __init__(self, size: torch.Size):
    """Builds a new LinearGauss structure.

    Args:
        size (torch.Size): This size must be broadcastable towards the later used
          input tensor.
    """
    super(LinearGauss, self).__init__()

    self.size = size
    self.mean = nn.Parameter(torch.zeros(size), dtype=torch.float16)
    self.lowStd = nn.Parameter(torch.zeros(size), dtype=torch.float16)
    self.highStd = nn.Parameter(torch.zeros(size), dtype=torch.float16)

  def forward(self, x: torch.Tensor):
    return irregularGauss(x=x, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)


class Lissajous(nn.Module):
  """
  Holds a Lissajous-like curve to be used as a sort of activation layer as a unit
    of knowledge.
  """
  def __init__(self, size: int):
    """Builds a new Lissajous-like curve structure.

    Args:
        size (int): The amount of dimensions encoded in the curve.
    """
    super(Lissajous, self).__init__()

    self.size = size
    self.frequency = nn.Parameter(torch.zeros([1, size]), dtype=torch.float16)
    self.phase = nn.Parameter(torch.zeros([1, size]), dtype=torch.float16)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Gets a sample or batch of samples from the contained curve.

    Args:
        x (torch.Tensor): The sample or sampling locations.

    Returns:
        torch.Tensor: The evaluted samples.
    """
    # Add another dimension to do the batch of encodes
    xFat = torch.unsqueeze(x, -1)
    xOnes = torch.ones_like(xFat)

    # Activate inside of the curve's embedding space
    cosinePosition = (xFat @ self.frequency) + (xOnes @ self.phase)
    evaluated = torch.cos(cosinePosition)

    return evaluated


class Knot(nn.Module):
  """
  Creates a Lissajous-Knot-like structure for encoding information. All information
    stored in the knot is stored in the form of a multidimensional fourier series,
    which allows the knot to have its parameters later entangled, modulated, and
    transformed through conventional methods.
  """
  def ___init__Helper(self, lissajousCurves: nn.ModuleList):
    """Does the actual __init__ work for super() call reasons.

    Args:
        lissajousCurves (nn.ModuleList): The curves to add together to create the
          knot.
    """
    # Set up the curves for the function
    self.curves = lissajousCurves
    self.curveSize = self.curves[0].size

    # Size assertion
    for curve in self.curves:
      assert curve.size == self.curveSize

    paramSize = (len(self.curves), self.curveSize)
    self.regWeights = nn.Parameter(torch.ones(paramSize), dtype=torch.float16)
    self.knotRadii = nn.Parameter(torch.zeros(self.curveSize), dtype=torch.float16)

  def __init__(self, lissajousCurves: nn.ModuleList):
    """Constructs a Knot for later use from previously constructed Lissajous curves.

    Args:
        lissajousCurves (nn.ModuleList): The Lissajous curves to add together to make the knot.
    """
    super(Knot, self).__init__()

    # Call helper init function
    self.___init___Helper(lissajousCurves=lissajousCurves)    

  def __init__(self, knotSize: int, knotDepth: int):
    """Constructs a Knot for later use generating all weights and storing internally.

    Args:
        knotSize (int): The dimensionality of the contained lissajous-like curves.
        knotDepth (int): The amount of lissajous-like curves to be added together.
    """
    super(Knot, self).__init__()

    # Construct and call helper function
    curves = nn.ModuleList([Lissajous(size=knotSize) for _ in range(knotDepth)])
    self.___init___Helper(lissajousCurves=curves)

  # TODO: Add a method to add more curves, it would be cool to have a hyperparameter
  #   that makes the neural network hold more data in almost the same space

  @torch.jit.script
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Pushed forward the same way as the Lissajous module. This is just an array
    of Lissajous modules summed together in a weighted way.

    Args:
        x (torch.Tensor): The points to sample on the curves.

    Returns:
        torch.Tensor: The orginal size tensor, but every point has a Lissajous curve
          activated upon it. There will be one extra dimension that is the same in size
          as the dimensions of the curve.
    """
    # Create the expanded dimensions required in the output tensor
    outputSize = list(x.size())
    outputSize = outputSize.append(self.curveSize)
    result = torch.Tensor(torch.zeros(outputSize), dtype=torch.float16)

    # Add all of the curves together
    for idx, lissajous in enumerate(self.curves):
      # Each lissajous curve-like structure has different weights, and therefore 
      curve = lissajous.forward(x)
      curve = self.regWeights[idx] * curve
      result = result + curve
    
    return result + self.knotRadii


# TODO: Continue adding size safety from [ HERE MARK SAFETY SIZES ]


# Entangle a whole bunch of knots into one singular signal
class KnotEntangle(nn.Module):
  def __init__(self, knots:nn.ModuleList, samples:int = DEFAULT_FFT_SAMPLES, lowerSmear:float = 1./8,
    upperSmear:float = 1./8, attn:bool = True, linearPolarization:bool = False, shareSmears:bool = False):
    """Generates the complex required to entangle two seperate knotted signals together.

    Args:
        knots (nn.ModuleList): The knots that define the array to be entangled into one knotted signal.
        samples (int, optional): The amount of FFT samples to use. Defaults to DEFAULT_FFT_SAMPLES.
        lowerSmear (float, optional): The proportion of the input that is smeared backwards. Defaults to 1./8.
        upperSmear (float, optional): The proportion of the input that is smeared forwards. Defaults to 1./8.
        attn (bool, optional): Try to make the input values more represented in the output signal. Defaults to True.
        linearPolarization (bool, optional): Embed a the entangled values with an elementwise knowledgegraph. Defaults to False.
        shareSmears (bool, optional): Share the smear windows between the knots. Defaults to False.
    """
    super(KnotEntangle, self).__init()

    # Set up the knots and assert size constraints
    self.knots = knots
    tCurveSize = self.knots[0].curveSize
    for knot in self.knots:
      assert knot.curveSize == tCurveSize

    # Define FFT and IFFT lead up and execution
    self.samples = samples
    if shareSmears:
      lowerProto = lowerSmear * torch.ones(len(self.knots))
      upperProto = upperSmear * torch.ones(len(self.knots))
      self.smearWindow = nn.Parameter(torch.Tensor([lowerProto, upperProto]), dtype=torch.float16)
    else:
      self.smearWindow = nn.Parameter(torch.Tensor([lowerSmear, upperSmear]), dtype=torch.float16)

    # Provide signal entanglement weighting
    self.entangleActivation = [LinearGauss(1.) for _ in range(self.knots)]
    self.entanglePolarization = nn.Parameter(torch.zeros(len(self.knots)), dtype=torch.float16)

    # If defined, this turns the entanglement function into something that is
    # initially, essentially, a dot product. This knowledge graph on the entanglemeant structure is
    # done per knot, and is referenced from the external entangling knot (as opposed to the
    # local entangled knot).
    self.linPolarization = linearPolarization
    if self.linPolarization:
      self.polKnowledge = nn.Parameter([torch.eye(self.samples) for _ in self.knots], dtype=torch.complex32)

    # Try to pay attention to the input values more than anything, adding some light weighting
    self.attn = attn
    if self.attn:
      self.attnWeight = nn.Parameter(torch.ones(len(self.knots)), dtype=torch.float16)
      self.attnBias = nn.Parameter(torch.zeros(len(self.knots)), dtype=torch.float16)
      self.attnScope = nn.Parameter(torch.ones(len(self.knots)), dtype=torch.float16)

  def knotCount(self) -> int:
    """Gets the amount of knots locked into the entanglement structure.

    Returns:
        int: The amount of knots to be locked into entanglement.
    """
    return len(self.knots)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Verify sizing
    ###

    # Create standardized sampling locations
    lowerSmear = self.smearWindow[0]
    upperSmear = self.smearWindow[1]
    xRange = (upperSmear - lowerSmear) * x
    xStep = xRange / self.samples
    xLow = ((1 - lowerSmear) * x)
    xIter = torch.Tensor([(builder + 1) / self.samples for builder in range(self.samples)], dtype=torch.float16).detach()

    # Smear sample
    knotSmears = []
    knotSignals = []
    for idx, knot in enumerate(self.knots):
      smear = knot.forward((xStep[idx] * xIter) + xLow[idx])
      knotSmears.append(smear)
      knotSignals.append(torch.fft.fft(smear))

    # Entangle
    entangledSmears = []
    for idx, signal in enumerate(knotSignals):
      # Find the entanglements
      smear = knotSmears[idx]
      signal = knotSignals[idx]
      resultSmear = torch.zeros_like(smear)

      for jdx in range(len(self.knots)):
        if idx == jdx: continue

        # Check signal correlation
        subsig = knotSignals[jdx]
        correlation = torch.mean(torch.fft.ifft(signal * torch.conj(subsig)))

        # Entangle signals
        # Note that the weighted activations are tied to each target knot
        entangleMix = self.entangleActivation[jdx].forward(correlation)
        classicalMix = 1 - entangleMix

        # Basing the entangling process of off the use of a tensor product mixed
        # with a sum. To collapse each entangled state, the view from each particle is
        # assessed and the more important one is superimposed into the final signal.
        superposition = (subsig @ torch.transpose(signal)) * self.polKnowledge
        collapseSignal = (torch.sum(superposition), torch.sum(torch.transpose(superposition)))
        collapseSmear = (torch.ifft(collapseSignal[0]), torch.ifft(collapseSignal[1]))
        polarization = self.entanglePolarization[jdx]
        entangledSmear = (torch.cos(polarization) * collapseSmear[0]) \
          + (torch.sin(polarization) * collapseSmear[1])

        # Mix the signals together and ensure normalization for what is entangled.
        resultSmear = resultSmear + (entangleMix * entangledSmear)
        resultSmear = resultSmear + (classicalMix * smear)

      # Push to end of calculation
      entangledSmears.append(resultSmear)

    # Collapse into a single knotted time-domain signal definition
    result = torch.sum(entangledSmears)

    # Don't pay attention if that's how you roll
    if not self.attn:
      return result

    # Mix the signal with a gaussian curve to signify the original importance of x 
    # if obscured. The idea is to use this as a way to pay attention to a specific
    # portion of the curve.
    allMeans = (x * self.attnWeight) + self.attnBias
    meansMean = torch.mean(x)
    allLows = (1. - (lowerSmear * self.attnScope)) * meansMean 
    allHighs = (1. + (upperSmear * self.attnScope)) * meansMean
    gaussSamples = ((allHighs - allLows) * xIter) + allLows
    gaussians = []
    for idx in range(len(self.knots)):
      # Pull out specific gaussian parameters
      activeMean = allMeans[idx]
      activeMeans = activeMean * torch.ones_like(result)
      gaussians.append(irregularGauss(x=gaussSamples, mean=activeMeans, lowStd=allLows, highStd=allHighs))
    
    # Apply the psuedo-attention and return
    return torch.sum(gaussians) * result


class KnotConv(nn.Module):
  def __init__(self, knots:nn.ModuleList = None, windowSize:tuple = (32, 32), stepSize:int = 4, samples:int = 128, knotSize:int = 3):
    super(KnotConv, self).__init__()

    self.windowSize = torch.Tensor(windowSize)
    self.stepSize = stepSize * torch.ones_like(self.windowSize)
    self.samples = samples
    self.knotSize = knotSize

    flatWindow = 1
    for n in windowSize:
      flatWindow = flatWindow * n

    self.knots = knots
    if self.knots != None:
      assert len(self.knots) == flatWindow
      for knot in self.knots:
        assert knot.curveSize == self.knotSize
    else:
      self.knots = nn.ModuleList([Knot(knotSize=knotSize, knotDepth=samples/4.) for i in range(flatWindow)])
