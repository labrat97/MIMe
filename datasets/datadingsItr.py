import os
import wget

def loadSourcesBundle(path):
    """Load Sources to a Bundle
    
    Loads the full set of links needed to download a bundled set of sources into
    a tuple described by the return section. This does not retrieve the sources.

    Args:
        path (string): The path to the bundle of data.

    Returns:
        Tuple(string, Dict(string, List(string))): The path, sources, and lists 
        of links each source respectively requires.
    """

    # Result
    sources = {}

    # Open the bundle from the provided path
    with open(path) as f:
        # Read the full descriptor
        lines = f.readlines()
        recentLabel = ''

        # Interprit
        for line in lines:
            # Found dictionary key
            if line[0] == '[':
                endItr = line.find(']:')
                recentLabel = line[1:endItr]
                if not recentLabel in sources.keys():
                    sources[recentLabel] = []

            # Found comment
            elif line[:2].lower() == '\"\"'\
                or line.replace(' ', '').strip() == '':
                continue

            # Found source
            else:
                sources[recentLabel].append(line.strip().lower())

    return (path, sources)

def retrieveSources(bundle, verbose=True):
    """Retrieve the bundled Sources using wget.

    Using the bundle type described, optionally display, and download all files
    provided using the wget python plugin.

    Args:
        bundle (Tuple(string, Dict(string, List(string)))): The bundle of sources
            that needs to be downloaded.
        verbose (bool, optional): Display what is being done. Defaults to True.

    Returns:
        Dict(string, List(string)): A dictionary of the downloaded dataset names
            and the paths of the files that were downloaded.
    """
    if verbose:
        print('Retrieving sources...')

    # Extract info
    path, sources = bundle
    dumpPath = os.path.dirname(path)

    # Prepare result
    downloadPaths = {}
    
    # Download each source file
    for source in sources.keys():
        downloadPaths[source] = []

        # Gain access to directory if not already present
        subDump = os.path.join(dumpPath, source).upper()
        if not os.path.exists(subDump):
            os.makedirs(subDump)
        
        # Download using wget
        for link in sources[source]:
            print(link)
            try:
                downloadPaths[source].append( \
                    wget.download(link, out=os.path.join(subDump, '.'), \
                    bar=wget.bar_adaptive if verbose else None))
            except Exception:
                print(".Fuck.")

            print('')

    return downloadPaths

def downloadAll(verbose=True):
    # Get the topmost path to search
    top = os.path.dirname(__file__)

    # Walk through the directory tree
    for dirname, _, files in os.walk(top):
        for file in files:
            if file[-5:].lower() != '.srcs': continue

            # Display match if required
            if verbose:
                print(f'Sources found at: \"{file}\"')

            # Download associated files
            srcBundle = loadSourcesBundle(os.path.join(dirname, file))
            _ = retrieveSources(srcBundle, verbose=verbose)

