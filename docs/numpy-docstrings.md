# NumPy-style docstrings

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
