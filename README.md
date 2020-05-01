# Graph MBO

[![PyPI Version](https://img.shields.io/pypi/v/graph-mbo.svg)](https://pypi.org/project/graph-mbo/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/graph-mbo.svg)](https://pypi.org/project/graph-mbo/)
[![Build Status](https://github.com/ephesosx/graph-mbo/workflows/CI/badge.svg)](https://github.com/ephesosx/graph-mbo/actions)
[![Documentation](https://readthedocs.org/projects/graph-mbo/badge/?version=stable)](https://graph-mbo.readthedocs.io/en/stable/?badge=stable)
[![Code Coverage](https://codecov.io/gh/ephesosx/graph-mbo/branch/master/graph/badge.svg)](https://codecov.io/gh/ephesosx/graph-mbo)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An implementation of the MBO scheme on graphs.

---

## Installation

To install Graph MBO, run this command in your terminal:

```bash
$ pip install -U graph-mbo
```

This is the preferred method to install Graph MBO, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## Quick Start
```python
>>> from graph_mbo import Example
>>> a = Example()
>>> a.get_value()
10

```

## Citing
If you use our work in an academic setting, please cite our paper:


## Documentation
TODO: readthedocs
For more information, read the docs.


## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

#### Suggested Git Branch Strategy
1. `master` is for the most up-to-date development, very rarely should you directly commit to this branch. Your day-to-day work should exist on branches separate from `master`. It is recommended to commit to development branches and make pull requests to master.3. Even if it is just yourself working on the repository, make a pull request from your working branch to `master` so that you can ensure your commits don't break the development head. GitHub Actions will run on every push to any branch or any pull request from any branch to any other branch.4. It is recommended to use "Squash and Merge" commits when committing PR's. It makes each set of changes to `master`
atomic and as a side effect naturally encourages small well defined PR's.


#### Additional Optional Setup Steps:
* Create an initial release to test.PyPI and PyPI.
    * Follow [This PyPA tutorial](https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives), starting from the "Generating distribution archives" section.

* Create a blank github repository (without a README or .gitignore) and push the code to it.

* Create an account on [codecov.io](https://codecov.io/) and link it with your GitHub account. Code coverage should be updated automatically when you commit to `master`.
* Add branch protections to `master`
    * Go to your [GitHub repository's settings and under the `Branches` tab](https://github.com/ephesosx/graph-mbo/settings/branches), click `Add rule` and select the
    settings you believe best.
    * _Recommendations:_
      * _Require status checks to pass before merging_

* Setup readthedocs. Create an account on [readthedocs.org](https://readthedocs.org/) and link it to your GitHub account.
    * Go to your account page and select "Import a Project".
    * Select the desired GitHub repository from the list, refreshing first if it is not present.
    * Go to the admin panel of the new project and make some changes to the "advanced settings":
        * Enable "Show version warning"
        * Enter "rtd-reqs.txt" into the "Requirements file" field
        * Enable "Install Project"
        * Enable "Use system packages"
        * Make sure to click save at the bottom when you are finished editing the settings

* Delete these setup instructions from `README.md` when you are finished with them.
