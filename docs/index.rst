.. chandra_aca documentation master file, created by
   sphinx-quickstart on Fri Nov 13 15:19:28 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

chandra_aca
===========

The chandra_aca package provides functionality related to the Chandra Aspect
Camera Assembly and associated aspect functionality.

.. toctree::
   :maxdepth: 2

Plotting
--------

.. automodule:: chandra_aca.plot
   :members:

Star probabilities
------------------

.. automodule:: chandra_aca.star_probs
   :members:

Transformations
---------------

.. automodule:: chandra_aca.transform
   :members:

Centroid residuals
------------------

.. automodule:: chandra_aca.centroid_resid
   :members:

ACA alignment drift
-------------------

.. automodule:: chandra_aca.drift
   :members:

ACA images
----------

The `aca_image` module contains classes and utilities related to ACA readout
images.  This includes the `ACAImage class`_ for manipulating images in
ACA coordinates, a first moment centroiding routine, and a library for
generating synthetic ACA images that have a high-fidelity point spread function
(PSF).

.. automodule:: chandra_aca.aca_image
   :members:

ACAImage class
^^^^^^^^^^^^^^

ACAImage is an ndarray subclass that supports functionality for the Chandra
ACA. Most importantly it allows image indexing and slicing in absolute
"aca" coordinates, where the image lower left coordinate is specified
by object ``row0`` and ``col0`` attributes.

It also provides a ``meta`` dict that can be used to store additional useful
information.  Any keys which are all upper-case will be exposed as object
attributes, e.g. ``img.BGDAVG`` <=> ``img.meta['BGDAVG']``.  The ``row0``
attribute  is a proxy for ``img.meta['IMGROW0']``, and likewise for ``col0``.

Creation
""""""""

When initializing an ``ACAImage``, additional ``*args`` and ``**kwargs`` are
used to try initializing via ``np.array(*args, **kwargs)``.  If this fails
then ``np.zeros(*args, **kwargs)`` is tried.  In this way one can either
initialize from array data or create a new array of zeros.

One can easily create a new ``ACAImage`` as shown below.  Note that it must
always be 2-dimensional.  The initial ``row0`` and ``col0`` values default
to zero.
::

  >>> from chandra_aca.aca_image import ACAImage
  >>> im4 = np.arange(16).reshape(4, 4)
  >>> a = ACAImage(im4, row0=10, col0=20)
  >>> a
  <ACAImage row0=10 col0=20
  array([[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15]])>

One could also initialize by providing a ``meta`` dict::

  >>> a = ACAImage(im4, meta={'IMGROW0': 10, 'IMGCOL0': 20, 'BGDAVG': 5.2})
  >>> a
  <ACAImage row0=10 col0=20
  array([[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15]])>

.. Note::

   The printed representation of an ``ACAImage`` is always shown as the
   rounded integer version of the values, but the full float value
   is stored internally.  If you ``print()`` the image then the floating
   point values are shown.

Image access and setting
""""""""""""""""""""""""

You can access array elements as usual, and in fact do any normal numpy array operations::

  >>> a[3, 3]
  15

The special nature of ``ACAImage`` comes by doing array access via the ``aca`` attribute.
In this case all index values are in absolute coordinates, which might be negative.  In
this case we can access the pixel at ACA coordinates row=13, col=23, which is equal to
``a[3, 3]`` for the given ``row0`` and ``col0`` offset::

  >>> a.aca[13, 23]
  15

Creating a new array by slicing adjusts the ``row0`` and ``col0`` values
like you would expect::

  >>> a2 = a.aca[12:, 22:]
  >>> a2
  <ACAImage row0=12 col0=22
  array([[500,  11],
         [ 14,  15]])>

You can set values in absolute coordinates::

  >>> a.aca[11:13, 21:23] = 500
  >>> a
  <ACAImage row0=10 col0=20
  array([[  0,   1,   2,   3],
         [  4, 500, 500,   7],
         [  8, 500, 500,  11],
         [ 12,  13,  14,  15]])>

Now let's make an image that represents the full ACA CCD and set a
sub-image from our 4x4 image ``a``.  This uses the absolute location
of ``a`` to define a slice into ``b``::

  >>> b = ACAImage(shape=(1024,1024), row0=-512, col0=-512)
  >>> b[a] = a
  >>> b.aca[8:16, 18:26]
  <ACAImage row0=8 col0=18
  array([[  0,   0,   0,   0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0,   0,   0,   0],
         [  0,   0,   0,   1,   2,   3,   0,   0],
         [  0,   0,   4, 500, 500,   7,   0,   0],
         [  0,   0,   8, 500, 500,  11,   0,   0],
         [  0,   0,  12,  13,  14,  15,   0,   0],
         [  0,   0,   0,   0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0,   0,   0,   0]])>

You can also do things like adding 100 to every pixel in ``b``
within the area of ``a``::

  >>> b[a] += 100
  >>> b.aca[8:16, 18:26]
  <ACAImage row0=8 col0=18
  array([[  0,   0,   0,   0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0,   0,   0,   0],
         [  0,   0, 100, 101, 102, 103,   0,   0],
         [  0,   0, 104, 600, 600, 107,   0,   0],
         [  0,   0, 108, 600, 600, 111,   0,   0],
         [  0,   0, 112, 113, 114, 115,   0,   0],
         [  0,   0,   0,   0,   0,   0,   0,   0],
         [  0,   0,   0,   0,   0,   0,   0,   0]])>

Image arithmetic operations
"""""""""""""""""""""""""""

In addition to doing image arithmetic operations using explicit slices
as shown previously, one can also use normal arithmetic operators like
``+`` (for addition) or ``+=`` for in-place addition.

**When the right-side operand is included via its ``.aca`` attribute, then the operation is
done in ACA coordinates.**

This means that the operation is only done on overlapping pixels.  This is
shown in the examples below.  The supported operations for this are:

- Addition (``+`` and ``+=``)
- Subtraction (``-`` and ``-=``)
- Multiplication (``*`` and ``*=``)
- Division (``/`` and ``/=``)
- True division (``/`` and ``/=`` in Py3+ and with __future__ division)
- Floor division (``//`` and ``//=``)
- Modulus (``%`` and ``%=``)
- Power (``**`` and ``**=``)

**Initialize images (different shape and offset)**

  >>> a = ACAImage(shape=(6, 6), row0=10, col0=20) + 1
  >>> a
  <ACAImage row0=10 col0=20
  array([[1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1]])>
  >>> b = ACAImage(np.arange(1, 17).reshape(4, 4), row0=8, col0=18) * 10
  >>> b
  <ACAImage row0=8 col0=18
  array([[ 10,  20,  30,  40],
         [ 50,  60,  70,  80],
         [ 90, 100, 110, 120],
         [130, 140, 150, 160]])>

**Add images (output has shape and row0/col0 of left side input)**

  >>> a + b.aca
  <ACAImage row0=10 col0=20
  array([[111, 121,   1,   1,   1,   1],
         [151, 161,   1,   1,   1,   1],
         [  1,   1,   1,   1,   1,   1],
         [  1,   1,   1,   1,   1,   1],
         [  1,   1,   1,   1,   1,   1],
         [  1,   1,   1,   1,   1,   1]])>
  >>> b + a.aca
  <ACAImage row0=8 col0=18
  array([[ 10,  20,  30,  40],
         [ 50,  60,  70,  80],
         [ 90, 100, 111, 121],
         [130, 140, 151, 161]])>

  >>> b += a.aca
  >>> b
  <ACAImage row0=8 col0=18
  array([[ 10,  20,  30,  40],
         [ 50,  60,  70,  80],
         [ 90, 100, 111, 121],
         [130, 140, 151, 161]])>

**Make ``b`` image be fully contained in ``a``**

  >>> b.row0 = 11
  >>> b.col0 = 21
  >>> a += b.aca
  >>> a
  <ACAImage row0=10 col0=20
  array([[  1,   1,   1,   1,   1,   1],
         [  1,  11,  21,  31,  41,   1],
         [  1,  51,  61,  71,  81,   1],
         [  1,  91, 101, 112, 122,   1],
         [  1, 131, 141, 152, 162,   1],
         [  1,   1,   1,   1,   1,   1]])>

**Normal image addition fails if shape is mismatched**

  >>> a + b
  Traceback (most recent call last):
    File "<ipython-input-19-f96fb8f649b6>", line 1, in <module>
      a + b
    File "chandra_aca/aca_image.py", line 68, in _operator
      out = op(self, other)  # returns self for inplace ops
  ValueError: operands could not be broadcast together with shapes (6,6) (4,4)


Meta-data
"""""""""

Finally, the ``ACAImage`` object can store arbitrary metadata in the
``meta`` dict attribute.  However, in order to make this convenient and
distinct from native numpy attributes, the ``meta`` attributes should
have UPPER CASE names.  In this case they can be directly accessed
as object attributes instead of going through the ``meta`` dict::

  >>> a.IMGROW0
  10
  >>> a.meta
  {'IMGCOL0': 20, 'IMGROW0': 10}
  >>> a.NEWATTR = 'hello'
  >>> a.meta
  {'IMGCOL0': 20, 'NEWATTR': 'hello', 'IMGROW0': 10}
  >>> a.NEWATTR
  'hello'
  >>> a.meta['fail'] = 1
  >>> a.fail
  Traceback (most recent call last):
  AttributeError: 'ACAImage' object has no attribute 'fail'



Maude Decom
-----------

.. automodule:: chandra_aca.maude_decom
   :members:
