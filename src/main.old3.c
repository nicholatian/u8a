/****************************************************************************\
 *                                   u8a.                                   *
 *                                                                          *
 *                   Copyright (C) 2021 Alexander Nicholi                   *
 *                           All rights reserved.                           *
\****************************************************************************/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "numpy/ufuncobject.h"
#include <stdio.h>
#include <stdlib.h>

/* BEGIN types.h */

#if !defined( NULL )
#if defined( __cplusplus )
#define NULL ( (void *)0 )
#else
#define NULL 0
#endif /* defined( __cplusplus ) */
#endif /* !defined( NULL ) */

#if defined( __clang__ ) || defined( __GNUC__ )
#define UNI_HAS_I128( ) 1
#define UNI_HAS_I64( ) 1
#elif defined( __TINYC__ )
#define UNI_HAVE_I128( ) 0
#define UNI_HAVE_I64( ) 1
#endif /* defined( __clang__ ) || defined( __GNUC__ ) */

enum
{
	SIZEOF_CHAR = 1,
#if defined( __i386__ ) || defined( CFG_GBA )
	SIZEOF_PTR = 4,
#elif defined( __x86_64__ )
	SIZEOF_PTR = 8,
#endif
	S8_MIN  = -128,
	S8_MAX  = 127,
	S16_MIN = -32768,
	S16_MAX = 32767,
	U8_MAX  = 255,
	U16_MAX = 65535
};

enum
{
	S32_MIN = -2147483648,
	S32_MAX = 2147483647,
};

enum
{
	U32_MAX = 4294967295U
};

enum
{
	U64_MAX = 18446744073709551615UL
};

enum
{
#if defined( __i386__ ) || defined( CFG_GBA )
	PTRI_MAX = U32_MAX
#elif defined( __x86_64__ )
	PTRI_MAX   = U64_MAX
#else
#error "Unknown platform."
#endif
};

#if UNI_HAS_I128( )
#define U128_MAX ( ( ( U64_MAX + 1 ) * ( U64_MAX + 1 ) ) - 1 )
#define S128_MIN ( ( ( S64_MIN * -1 ) * ( S64_MIN * -1 ) ) * -1 )
#define S128_MAX ( ( ( S64_MAX + 1 ) * ( S64_MAX + 1 ) ) - 1 )
#endif /* UNI_HAS_I128( ) */

#ifdef _MSC_VER
typedef unsigned __int8 u8;
typedef unsigned __int16 u16;
typedef unsigned __int32 u32;
typedef unsigned __int64 u64;
typedef signed __int8 s8;
typedef signed __int16 s16;
typedef signed __int32 s32;
typedef signed __int64 s64;
typedef signed __int128 s128;
#ifdef _M_IX86
typedef unsigned __int32 ptri;
typedef __int32 offs;
#elif defined( _M_X64 )
typedef unsigned __int64 ptri;
typedef __int64 offs;
#else
#error "Must be compiling for i386 or AMD64 when targeting Windows"
#endif /* _M_ arch */
/* UNIX land */
#elif defined( CFG_GBA )
typedef __INT32_TYPE__ s32;
typedef __INT16_TYPE__ s16;
typedef __INT8_TYPE__ s8;
typedef __UINT32_TYPE__ u32;
typedef __UINT16_TYPE__ u16;
typedef __UINT8_TYPE__ u8;
typedef __UINT32_TYPE__ ptri;
typedef __INT32_TYPE__ offs;
#define UNI_HAS_I128( ) 0
#define UNI_HAS_I64( ) 0
#elif defined( __clang__ ) || defined( __GNUC__ )
typedef signed __int128 s128;
typedef __INT64_TYPE__ s64;
typedef __INT32_TYPE__ s32;
typedef __INT16_TYPE__ s16;
typedef __INT8_TYPE__ s8;
typedef unsigned __int128 u128;
typedef __UINT64_TYPE__ u64;
typedef __UINT32_TYPE__ u32;
typedef __UINT16_TYPE__ u16;
typedef __UINT8_TYPE__ u8;
typedef __UINTPTR_TYPE__ ptri;
typedef __INTPTR_TYPE__ offs;
#define UNI_HAS_I128( ) 1
#define UNI_HAS_I64( ) 1
#elif defined( __TINYC__ )
#include <stdint.h>
typedef int64_t s64;
typedef int32_t s32;
typedef int16_t s16;
typedef int8_t s8;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef uintptr_t ptri;
typedef intptr_t offs;
#define UNI_HAS_I128( ) 0
#define UNI_HAS_I64( ) 1
#endif

typedef u16 char16;
typedef u32 char32;

typedef volatile s8 vs8;
typedef volatile s16 vs16;
typedef volatile s32 vs32;
#if UNI_HAS_I64( )
typedef volatile s64 vs64;
#endif /* UNI_HAS_I64( ) */
#if UNI_HAS_I128( )
typedef volatile s128 vs128;
#endif /* UNI_HAS_I128( ) */

typedef volatile u8 vu8;
typedef volatile u16 vu16;
typedef volatile u32 vu32;
#if UNI_HAS_I64( )
typedef volatile u64 vu64;
#endif /* UNI_HAS_I64( ) */
#if UNI_HAS_I128( )
typedef volatile u128 vu128;
#endif /* UNI_HAS_I128( ) */

typedef volatile ptri vptri;
typedef volatile offs voffs;

typedef volatile char vchar;
typedef volatile char16 vchar16;
typedef volatile char32 vchar32;

/* END types.h */

struct u8a
{
	ptri sz;
	u8 * data;
};

struct u8a u8a_add( struct u8a a, struct u8a b );

/* This declares the wrapper function */
static void py_u8a_add( char **, const npy_intp * , const npy_intp *, void * );

static const char k_u8aadd_docstring[] =
	"Calculate the covariances between a vector and a list of vectors.";

/*
 * This tells Python what methods this module has.
 * See the Python-C API for more information.
 */
static const struct PyMethodDef k_u8a_methods[] = {
	{ NULL, NULL, 0, NULL } };

PyUFuncGenericFunction funcs[1] = {&py_u8a_add};

/* This initiates the module using the above definitions. */
static const struct PyModuleDef k_module_def = { PyModuleDef_HEAD_INIT,
	"npufunc",
	NULL,
	-1,
	(struct PyMethodDef *)k_u8a_methods,
	NULL,
	NULL,
	NULL,
	NULL };

PyObject * PyInit_u8a( void )
{
	PyObject * ret = PyModule_Create( (struct PyModuleDef *)&k_module_def );

	if( ret == NULL )
	{
		return ret;
	}

	import_array( );

	return ret;
}

static void py_u8a_add( char ** args, const npy_intp * dims, const npy_intp  strides, void * inner_arr_data )
{
}

static PyObject * py_u8a_add( PyObject * self, PyObject * args )
{
	PyObject * ret   = NULL;
	PyObject * a_obj = NULL;
	PyObject * b_obj = NULL;
	PyObject * a_arr = NULL;
	PyObject * b_arr = NULL;
	npy_intp sz;

	if( !PyArg_ParseTuple( args, "OO", &a_obj, &b_obj ) )
	{
		return NULL;
	}

	a_arr = PyArray_FROM_OTF( a_obj, NPY_UBYTE, NPY_IN_ARRAY );

	if( a_arr == NULL || PyArray_NDIM( a_arr ) != 1 )
	{
		Py_XDECREF( a_arr );

		return NULL;
	}

	b_arr = PyArray_FROM_OTF( b_obj, NPY_UBYTE, NPY_IN_ARRAY );

	if( b_arr == NULL || PyArray_NDIM( b_arr ) != 1 )
	{
		Py_XDECREF( b_arr );

		return NULL;
	}

	sz = PyArray_SIZE( a_arr );

	if( PyArray_SIZE( b_arr ) != sz )
	{
		Py_XDECREF( a_arr );
		Py_XDECREF( b_arr );

		return NULL;
	}

	{
		struct u8a a, b, r;
		const npy_intp dimensions[1] = { 1 };

		a.data = (u8*)PyArray_BYTES( a_arr );
		b.data = (u8*)PyArray_BYTES( b_arr );
		a.sz   = sz;
		b.sz   = sz;

		r = u8a_add( a, b );

		ret = PyArray_SimpleNew( 1, dimensions, NPY_UBYTE );
		memcpy( PyArray_DATA( ret ), r.data, sz * sizeof( u8 ) );

		free( r.data );
		r.data = NULL;
	}

	return ret;
}

struct u8a u8a_add( struct u8a a, struct u8a b )
{
	struct u8a ret = { 0, NULL };
	const ptri sz  = a.sz;
	u16 m;
	ptri i;

	ret.data = calloc( sz, sizeof( u8 ) );

	if( ret.data == NULL )
	{
		return ret;
	}

	ret.sz = sz;

	for( i = 0; i < sz; ++i )
	{
		m = a.data[i];
		m += b.data[i];

		ret.data[i] = m > 0xFF ? 0xFF : m;
	}

	return ret;
}
