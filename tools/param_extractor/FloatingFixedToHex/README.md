## Simple tool to convert floating point/fixed point number to HEX for Python

This is a small Python script to transform a floating point number to HEX 
form of its floating point or fixed point number. The format of the fixed point 
can be defined.

### Requirement:

It requires _Simple Python Fixed-Point Module (SPFPM)_ . It can be installed by

```
pip install spfpm
```

### Usage

Import the Python file:

```python
>>> import ffth
```

...and use the functions as you want:
```python
>>> print(ffth.float_to_hex(5.28))
40a8f5c3
>>> # ffth.float_to_fixed_hex(data, number_of_fractional_bits, number_of_signed_intger_bits)
>>> print(ffth.float_to_fixed_hex(-123.28, 8, 8))
84b9
>>> print(ffth.hex_to_float('deadbeef'))
-6.259853398707798e+18
```
