# Images

Read a single channel from a Gwyddion file with `GwyReader`, wrap it in an
`Image`, and apply a chain of transformations such as `LineLevel` and
`BackgroundPolyFit`:

```python
from pySNOM.readers import GwyReader
from pySNOM.images import Image, LineLevel, BackgroundPolyFit, DataTypes

reader = GwyReader(fullfilepath="measurement.gwy", channelname="O3A raw")
gwy_data = reader.read()

image = Image(gwy_data, channelname="O3A raw")

leveled = LineLevel(method="median", datatype=DataTypes.Amplitude)
image.data = leveled.transform(image.data)

flattened = BackgroundPolyFit(xorder=1, yorder=1, datatype=DataTypes.Amplitude)
image.data = flattened.transform(image.data)
```

For stacks of images acquired at different wavenumbers (for example
hyperspectral scans), `pySNOM.images` also provides `AlignImageStack` and
related helpers (`create_nparray_stack`, `sort_image_stack`,
`dict_from_imagestack`) to align and reshape the stack before processing
each frame.

See the [API reference](../api/images.rst) for the full list of image
transformations available.
