#include <Python.h>
#include <numpy/arrayobject.h>
#include "add.h"
#include <stdio.h>

static char module_docstring[] =
	"This module provides an interface for calculating covariance using C.";

static char cov_vec_docstring[] =
	"Calculate the covariances between a vector and a list of vectors.";

static PyObject * _aux_covVec( PyObject * self, PyObject * args );

static PyMethodDef module_methods[] = {
	{ "cov_vec", _aux_covVec, METH_VARARGS, cov_vec_docstring },
	{ NULL, NULL, 0, NULL } };

PyMODINIT_FUNC init_aux( void )
{
	PyObject * m =
		Py_InitModule3( "_aux", module_methods, module_docstring );
	if( m == NULL )
		return;

	/* Load `numpy` functionality. */
	import_array( );
}

static PyObject * _aux_covVec( PyObject * self, PyObject * args )
{
	PyObject *X_obj, *x_obj;

	/* Parse the input tuple */
	if( !PyArg_ParseTuple( args, "OO", &X_obj, &x_obj ) )
		return NULL;

	/* Interpret the input objects as numpy arrays. */
	PyObject * X_array =
		PyArray_FROM_OTF( X_obj, NPY_DOUBLE, NPY_IN_ARRAY );
	PyObject * x_array =
		PyArray_FROM_OTF( x_obj, NPY_DOUBLE, NPY_IN_ARRAY );

	/* If that didn't work, throw an exception. */
	if( X_array == NULL || x_array == NULL )
	{
		Py_XDECREF( X_array );
		Py_XDECREF( x_array );
		return NULL;
	}

	/* What are the dimensions? */
	int nvecs  = (int)PyArray_DIM( X_array, 0 );
	int veclen = (int)PyArray_DIM( X_array, 1 );
	int xlen   = (int)PyArray_DIM( x_array, 0 );

	/* Get pointers to the data as C-types. */
	double * X = (double *)PyArray_DATA( X_array );
	double * x = (double *)PyArray_DATA( x_array );

	/* Call the external C function to compute the covariance. */
	double * k = covVec( X, x, nvecs, veclen );

	if( veclen != xlen )
	{
		PyErr_SetString(
			PyExc_RuntimeError, "Dimensions don't match!!" );
		return NULL;
	}

	/* Clean up. */
	Py_DECREF( X_array );
	Py_DECREF( x_array );

	int i;
	for( i = 0; i < nvecs; i++ )
	{
		printf( "k[%d]   = %f\n", i, k[i] );
		if( k[i] < 0.0 )
		{
			PyErr_SetString( PyExc_RuntimeError,
				"Covariance should be positive but it isn't." );
			return NULL;
		}
	}

	npy_intp dims[1] = { nvecs };

	PyObject * ret = PyArray_SimpleNew( 1, dims, NPY_DOUBLE );
	memcpy( PyArray_DATA( ret ), k, nvecs * sizeof( double ) );
	free( k );

	return ret;
}
