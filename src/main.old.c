/****************************************************************************\
 *                                   u8a.                                   *
 *                                                                          *
 *                   Copyright (C) 2021 Alexander Nicholi                   *
 *                           All rights reserved.                           *
\****************************************************************************/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

/* This declares the function */
static PyObject * u8a_add( PyObject * self, PyObject * args );

/*
 * This tells Python what methods this module has.
 * See the Python-C API for more information.
 */
static const struct PyMethodDef k_u8a_methods[] = {
	{ "logit", u8a_add, METH_VARARGS, "compute logit" },
	{ NULL, NULL, 0, NULL } };

/*
 * This actually defines the logit function for
 * input args from Python.
 */

static PyObject * u8a_add( PyObject * self, PyObject * args )
{
	double p;

	/* This parses the Python argument into a double */
	if( !PyArg_ParseTuple( args, "d", &p ) )
	{
		return NULL;
	}

	/* THE ACTUAL LOGIT FUNCTION */
	p = p / ( 1 - p );
	p = log( p );

	/* This builds the answer back into a python object */
	return Py_BuildValue( "d", p );
}

/* This initiates the module using the above definitions. */
static const struct PyModuleDef k_module_def = { PyModuleDef_HEAD_INIT,
	"spam",
	NULL,
	-1,
	k_u8a_methods,
	NULL,
	NULL,
	NULL,
	NULL };

PyObject * PyInit_u8a( void ) { return PyModule_Create( &k_module_def ); }
