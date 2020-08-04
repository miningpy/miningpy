# Contributing

This page is dedicated to outline where you should start with your question,
concern, feature request, or desire to contribute.


## Cloning the Source Repository

You can clone the source repository from `https://bitbucket.org/incitron/miningpy`
and install the latest version by running:

```bash
git clone https://bitbucket.org/incitron/miningpy/miningpy.git
cd miningpy
python -m pip install .
```

## Reporting Bugs

If you stumble across any bugs, crashes, or concerning quirks while using code
distributed here, please report it on the [issues page](https://bitbucket.org/incitron/miningpy/issues)
with an appropriate label so we can promptly address it.
When reporting an issue, please be overly descriptive so that we may reproduce
it. Whenever possible, please provide tracebacks, screenshots, and sample files
to help us address the issue.


## Feature Requests

We encourage users to submit ideas for improvements to PyVista code base!
Please create an issue on the [issues page](https://bitbucket.org/incitron/miningpy/issues)
with a *feature* component to suggest an improvement.
Please use a descriptive title and provide ample background information to help
the community implement that functionality. For example, if you would like a
reader for a specific file format, please provide a link to documentation of
that file format and possibly provide some sample files with screenshots to work
with. We will use the issue thread as a place to discuss and provide feedback.


## Licensing

All contributed code will be licensed under The MIT License found in the
repository. If you did not write the code yourself, it is your responsibility
to ensure that the existing license is compatible and included in the
contributed files or you can obtain permission from the original author to
relicense the code.


## Development Practices
This section provides a guide to how we conduct development in the MiningPy repository. 
Please follow the practices outlined here when contributing directly to this repository.

There are three general coding paradigms that we believe in:

1. **Make it intuitive**. Any new features should have
   intuitive naming conventions and explicit keyword arguments for users to
   make the bulk of the library accessible to novice users.

2. **Document everything!** At the least, include a docstring for any method
   or class added. Do not describe what you are doing but why you are doing
   it and provide a for simple use cases for the new features.

3. **Keep it tested**. We aim for a high test coverage. See
   testing for more details.

There are two important copyright guidelines:

4. Please do not include any data sets for which a license is not available
   or commercial use is prohibited. Those can undermine the license of
   the whole projects.

5. Do not use code snippets for which a license is not available (e.g. from
   stackoverflow) or commercial use is prohibited. Those can undermine
   the license of the whole projects.
   
### Contributing to pyvista through Bitbucket

To submit new code to pyvista, first fork the [MiningPy Bitbucket Repo](https://bitbucket.org/incitron/miningpy) 
and then clone the forked repository to your computer. Then, create a new branch based on the
[Branch Naming Conventions Section](#branch-naming-conventions) in your local repository.

Next, add your new feature and commit it locally. Be sure to commit
often as it is often helpful to revert to past commits, especially if
your change is complex.  Also, be sure to test often. See the
[Testing Section](#testing) below for automating testing.

When you are ready to submit your code, create a pull request by
following the steps in the [Creating a New Pull Request section](#creating-a-new-pull-request).


#### Coding Style

We adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/)
wherever possible, except that line widths are permitted to go beyond
79 characters to a max of 90 to 100 characters.

Outside of PEP 8, when coding please consider [PEP 20 -- The Zen of Python](https://www.python.org/dev/peps/pep-0020/).  
When in doubt:

```python
import this
```


#### Branch Naming Conventions

To streamline development, we have the following requirements for naming
branches. These requirements help the core developers know what kind of changes
any given branch is introducing before looking at the code.

- `fix/`: any bug fixes, patches, or experimental changes that are minor
- `feat/`: any changes that introduce a new feature or significant addition
- `doc/`: for any changes only pertaining to documentation
- `release/`: releases (see below)


#### Testing

After making changes, please test changes locally before creating a pull
request. The following tests will be executed after any commit or pull request,
so we ask that you perform the following sequence locally to track down any new
issues from your changes.

In the project root directory, run:
```bash
python -m pytest
```

If all tests pass, you should be good to make a pull request.


#### Creating a New Pull Request

Once you have tested your branch locally, create a pull request on
[MiningPy Bitbucket](https://bitbucket.org/incitron/miningpy) while merging to
master.

To ensure someone else reviews your code, at least one other member of
the pyvista contributors group must review and verify your code meets
our community's standards.  Once approved, if you have write
permission you may merge the branch.  If you don't have write
permission, the reviewer or someone else with write permission will
merge the branch and delete the PR branch.

Since it may be necessary to merge your branch with the current
release branch (see below), please do not delete your branch if it
is a `fix/` branch.


### Branching Model

This project has a branching model that enables rapid development of
features without sacrificing stability, and closely follows the 
[Trunk Based Development](https://trunkbaseddevelopment.com/) approach.

The main features of our branching model are:

- The `master` branch is the main development branch.  All features,
  patches, and other branches should be merged here.  While all PRs
  should pass all applicable CI checks, this branch may be
  functionally unstable as changes might have introduced unintended
  side-effects or bugs that were not caught through unit testing.
- There will be one or many `release/` branches based on minor
  releases (for example `release/0.24`) which contain a stable version
  of the code base that is also reflected on PyPi/.  Hotfixes from
  `fix/` branches should be merged both to master and to these
  branches.  When necessary to create a new patch release these
  release branches will have their `VERSION` updated and be
  tagged with a patched semantic version (e.g. `0.24.1`).  This
  triggers CI to push to PyPi, and allow us to rapidly push hotfixes
  for past versions of `MiningPy` without having to worry about
  untested features.
- When a minor release candidate is ready, a new `release` branch will
  be created from `master` with the next incremented minor version
  (e.g. `release/0.25`), which will be thoroughly tested.  When deemed
  stable, the release branch will be tagged with the version (`0.25.0`
  in this case), and if necessary merged with master if any changes
  were pushed to it.  Feature development then continues on `master`
  and any hotfixes will now be merged with this release.  Older
  release branches should not be deleted so they can be patched as
  needed.


