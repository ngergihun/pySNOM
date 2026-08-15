# Community

Is something missing? Have you spotted a bug? Read on for details on how to
raise feature requests, or contribute to `pySNOM`.

## Bugs, issues, features

We are constantly adding new features and fixing existing bugs in the package. Hower if you spot issues or you have new ideas don't hasitate to create an issue on the 
[pySNOM GitHub repository](https://github.com/Quasars/pySNOM/issues).

## Join and contribute

If you like coding and want to to maximize the impact of your own work, please join us to make pySNOM the most tremendous data processing package.
By joining the project you advance science and help other researchers. We believe that open-source project are the best way to build something great and useful. 

I hope you decided to join the development. If so, here is a little guidance how you can contribute to the code.

### 1. Create a local copy of the repository

- **Fork the repository.** 
  You will want a "fork" repository to base your changes. We follow the fork -> branch -> pull request -> merge workflow common on GitHub. See https://help.github.com/articles/fork-a-repo/ for more details.
  Go to the [pySNOM repository](https://github.com/Quasars/pySNOM) and click the "fork"
  button to create your own copy of the project.
- **Clone the repository.** Open a terminal in the directory where you'd like
  the project to be stored, then clone the project to your local computer:

  ```console
  git clone https://github.com/your-username/pySNOM.git
  ```

- **Link your local copy to the main repository.** Change the directory to
  your newly created local repository:

  ```console
  cd pySNOM
  ```

  Now, add the upstream repository:

  ```console
  git remote add upstream https://github.com/Quasars/pySNOM.git
  ```

  Then, `git remote -v` will show two remote repositories named:

  - `upstream`, which refers to the `pySNOM` repository
  - `origin`, which refers to your personal fork

- **Update your repository.** Make sure your local repository is up-to-date,
  by pulling the latest changes from upstream:

  ```console
  git checkout master
  git pull upstream master
  ```

### 2. Create a Python environment for development

Once you have a git checkout of the code, you will want some kind of virtual environment to keep your development separate from your regular pySNOM install (and other Python work/programs you may have). The follow describes how to do this using the Anaconda conda environment system, but can also apply to virtualenvs.

- **Create a fresh environment.** 
Run "Anaconda Prompt" or similar and:

    conda config --add channels conda-forge
    conda create --name="pysnom-dev"
    conda activate pysnom-dev

- **Install pySNOM.** In your clean environment, navigate to the pySNOM
  repository and install pySNOM and its dependencies in editable mode:


Navigate to your pySNOM directory, then install in development mode:

    pip install -e .

It is highly recommended to set up `pre-commit` hooks to automatically check your code quality before each commit:

    pre-commit install

- **Set up testing.** `pySNOM` uses the standard library `unittest` framework
  for its test suite. Your contributions won't be accepted unless all the
  tests pass, so check the tests run on your own computer before you submit
  your changes. If all went well, you should be able to run the tests:

    python -m unittest orangecontrib.spectroscopy.tests

  This should run the tests in `pySNOM/tests`, which should all pass
  successfully.

- **Set up documentation tools.** All the features in `pySNOM` should be
  documented, so if your edit adds a new feature, or changes how other users
  will interact with the package, we ask that you also add changes to the
  documentation to explain it. We use [`sphinx`](https://www.sphinx-doc.org/)
  to build our documentation. Install the documentation dependencies:

  ```console
  python -m pip install -r docs/requirements.txt
  ```

  You can then build the documentation with `sphinx-autobuild` to host it on
  a local server and preview any changes you make:

  ```console
  sphinx-autobuild docs docs/_build/html
  ```

  Or build a static copy with:

  ```console
  cd docs
  make html
  ```

### 3. Develop your contribution

- **Create a git branch for the feature you want to work on.** 
  Use a descriptive name such as `new-feature`. Try to keep your main branch up-to-date with the upstream repository and base the new branch on top of that:

  ```console
  git checkout -b new-feature
  ```

- **Commit changes locally as you progress.** Use `git add` and
  `git commit` with descriptive commit messages.
- **Create tests first.** All features in `pySNOM` should be tested to check
  they work. Create at least one test that will only pass once the new feature works
  correctly. Add your new test to the existing test suite in
  `pySNOM/tests`, then run the test suite:

  ```console
  python -m unittest pySNOM.tests
  ```
- **Add your changes to the package.** Make sure that you document your
  changes by adding or editing docstrings, following the style guide below.
- **Make your final checks.** If your changes are successful, when you run
  the test suite there should be no errors. Also check that all your code is
  formatted correctly (this should be done automatically when you commit
  your changes if you installed `pre-commit` as above), and that any changes
  are documented. Then you should be ready to submit to the main repository.

### 4. Submit your contribution

- **Push your changes back to your fork on GitHub**:

  ```console
  git push origin new-feature
  ```

- **Make a pull request.** Go to GitHub. The new branch will show up with a
  green Pull Request button. Make sure the title and message are clear,
  concise, and self-explanatory. Then click the button to submit it.

### 5. Review process

- **Automatic tests.** When you make a pull request, GitHub will run tests to
  check the suite of tests on different operating systems using several
  versions of Python. The tests must pass for us to merge your changes, so we
  recommend checking on your own computer before submitting a pull request.

- **The pySNOM team will review your pull request.** A pull request must be
  approved by at least one pySNOM team member before merging.

## Communication

It's a good idea to let the team know what you're working on before embarking,
especially for a large project. We prefer to discuss this in GitHub issues, so please
file one describing your contribution before getting too far in.

For large projects in particular, consider working in a public Pull Request marked
`[WIP]` (work-in-progress) and ask for feedback along the way.

## Docstring style

Sphinx uses the Napoleon extension to render NumPy-style Python docstrings as
structured documentation. New public classes and methods should follow this
pattern:

```python
def scale(values, factor=1.0):
    """Scale an array by a constant factor.

    Parameters
    ----------
    values : numpy.ndarray
        Values to scale.
    factor : float, optional
        Multiplicative factor, by default 1.0.

    Returns
    -------
    numpy.ndarray
        Scaled values.

    Raises
    ------
    ValueError
        If ``factor`` is not finite.

    Examples
    --------
    >>> scale(np.array([1.0, 2.0]), factor=2.0)
    array([2., 4.])
    """
```

Common sections include `Parameters`, `Returns`, `Raises`, `See Also`,
`Notes`, `References`, and `Examples`. Section names and indentation matter;
the headings should be followed by a line of hyphens.
