import torch
import torch.nn as nn

#https://pytorch.org/docs/stable/jit_language_reference.html
from typing import Dict, List, Tuple
import math

DEFAULT_FFT_SAMPLES = 128


@torch.jit.script
def irregularGuass(x: torch.Tensor, mean: torch.Tensor, lowStd: torch.Tensor, highStd: torch.Tensor) -> torch.Tensor:
  # Grab the correct side of the curve
  if x <= mean: std = lowStd
  else: std = highStd

  # Never hits 0 or inf., easy to take derivative
  std = torch.exp(std)

  # Calculate the gaussian curve
  top = torch.square(x - mean)
  bottom = torch.square(std)
  return torch.exp((-0.5) * (top / bottom))

class LinearGuass(nn.Module):
  def __init__(self, size: int):
    super(LinearGuass, self).__init__()

    self.size = size
    self.mean = nn.Parameter(torch.zeros(size), dtype=torch.float16)
    self.lowStd = nn.Parameter(torch.zeros(size), dtype=torch.float16)
    self.highStd = nn.Parameter(torch.zeros(size), dtype=torch.float16)

  def forward(self, x):
    # Size assertion
    assert x.size == (self.size)

    return irregularGuass(x=x, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)


# Creates an N dimensional lissajous-like curve for use in knot activation.
class Lissajous(nn.Module):
  def __init__(self, size: int):
    super(Lissajous, self).__init__()

    self.size = size
    self.frequency = nn.Parameter(torch.zeros(size), dtype=torch.float16)
    self.phase = nn.Parameter(torch.zeros(size), dtype=torch.float16)

  @torch.jit.script
  def forward(self, x: float):
    cosinePosition = (self.frequency * x) + self.phase
    evaluated = torch.cos(cosinePosition)

    return evaluated

# Create a Fourier Knot from a list of Lissajous curves.
class Knot(nn.Module):
  def ___init__Helper(self, lissajousCurves: nn.ModuleList):
    # Set up the curves for the function
    self.curves = lissajousCurves
    self.curveSize = self.curves[0].size

    # Size assertion
    for curve in self.curves:
      assert curve.size == self.curveSize

    self.regWeights = nn.Parameter(torch.ones(len(self.curves)), dtype=torch.float16)
    self.regBiases = nn.Parameter(torch.zeros(len(self.curves)), dtype=torch.float16)

  def __init__(self, lissajousCurves: nn.ModuleList):
    super(Knot, self).__init__()

    # Call helper init function
    self.___init___Helper(lissajousCurves=lissajousCurves)    

  def __init__(self, knotSize: int, knotDepth: int):
    super(Knot, self).__init__()

    # Construct and call helper function
    curves = nn.ModuleList([Lissajous(size=knotSize) for _ in range(knotDepth)])
    self.___init___Helper(lissajousCurves=curves)

# TODO: Add a method to add more curves, it would be cool to have a hyperparameter
#   that makes the neural network hold more data in almost the same space

  @torch.jit.script
  def forward(self, x: float):
    # Add all of the curves together
    y = torch.Tensor(torch.zeros(self.curveSize), dtype=torch.float16)
    for idx, lissajous in enumerate(self.curves):
      # Each lissajous curve-like structure has different weights, and therefore 
      curve = lissajous.forward(x)
      curve = (self.regWeights[idx] * curve) + self.regBiases[idx]
      y = y + curve
    
    return y


# Entangle a whole bunch of knots into one singular signal
class KnotEntangle(nn.Module):
  def __init__(self, knots: nn.ModuleList, outSamples: int = DEFAULT_FFT_SAMPLES, lowerSmear: float = 1./8, upperSmear: float = 1./8):
    super(KnotEntangle, self).__init()

    self.knots = knots
    tCurveSize = self.knots[0].curveSize
    for knot in self.knots:
      assert knot.curveSize == tCurveSize

    self.smearWindow = nn.Parameter(torch.Tensor([lowerSmear, upperSmear]), dtype=torch.float16)
    self.samples = outSamples

    self.entangleActivation = [LinearGuass(1.) for _ in range(self.knots)]
    self.entanglePolarization = nn.Parameter(torch.zeros(len(self.knots)), dtype=torch.float16)

  def knotCount(self) -> int:
    return len(self.knots)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Verify sizing
    assert len(x.shape) == 1
    assert x.shape == (len(self.knots))

    # Create standardized sampling locations
    lowerSmear = self.smearWindow[0]
    upperSmear = self.smearWindow[1]
    xRange = (upperSmear - lowerSmear) * x
    xStep = xRange / self.samples
    xLow = ((1 - lowerSmear) * x)
    xIter = torch.Tensor([builder / self.samples for builder in range(self.samples - 1)], dtype=torch.float16).detach()

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
        superposition = subsig @ torch.transpose(signal)
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

    # Mix the signal with a guassian curve to signify the original importance of x 
    # if obscured. The idea is to use this as a way to pay attention to a specific
    # portion of the curve.
    allMeans = x
    allLows = (1. - lowerSmear) * x
    allHighs = (1. + upperSmear) * x
    gaussians = []
    for idx in range(x.size()[0]):
      # Pull out specific gaussian parameters
      activeMean = allMeans[idx]
      activeMeans = activeMean * torch.ones_like(result)
      activeLow = allLows[idx]
      activeLows = activeLow * torch.ones_like(result)
      activeHigh = allHighs[idx]
      activeHighs = activeHigh * torch.ones_like(result)

      # Create the sampling space for the specific signal
      samples = ((activeHighs - activeLows) * xIter) + activeLows

      gaussians.append(irregularGuass(x=samples, mean=activeMeans, lowStd=activeLow, highStd=activeHigh))
    
    # Apply the psuedo-attention and return
    return torch.sum(gaussians) * result
