import sys

# for test deployments only
if '--testversion' in sys.argv:
    index = sys.argv.index('--testversion')
    sys.argv.pop(index)  # Removes the '--testversion'
    ver = sys.argv.pop(index)  # Returns the element after the '--testversion'

    # be careful! this is only intended for automated testing on azure pipelines - do not use this in production
    with open('miningpy/VERSION', 'r') as version_file:
        old_ver = str(version_file.readline())

    with open('miningpy/VERSION', 'w') as version_file:
        version_file.write(old_ver + '.' + ver)
