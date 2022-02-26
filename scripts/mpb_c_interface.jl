## wrapping c source code in `mpb/src` and library ("user interface") code in `mpb/mpb`
"""
code from `/github/mpb/src/util/sphere-quad.c`, now compiled to `/github/mpb/src/util/sphere-quad.so`
"""

"""
#define NQUAD3 50 /* use 50-point quadrature formula by default */

/**********************************************************************/
#define K_PI 3.141592653589793238462643383279502884197
#define NQUAD2 12
/**********************************************************************/

double sqr(double x) { return x * x; }

double dist2(double x1, double y1, double z1,
	     double x2, double y2, double z2)
    {
    return sqr(x1-x2) + sqr(y1-y2) + sqr(z1-z2);
    }

double min2(double a, double b) { return a < b ? a : b; }

/* sort the array to maximize the spacing of each point with the
   previous points */
void sort_by_distance(int n, double x[], double y[], double z[], double w[])
    {
        ...
    }
"""

mpb_dist = joinpath(homedir(),"github","mpb")
mpb_src = joinpath(mpb_dist,"src")
mpb_lib = joinpath(mpb_dist,"mpb")
mpb_src_util = joinpath(mpb_src,"util")
mpb_src_matrices = joinpath(mpb_src,"matrices")
mpb_src_matrixio = joinpath(mpb_src,"matrixio")
mpb_src_maxwell = joinpath(mpb_src,"maxwell")

const mpb_sphere_quad = joinpath(mpb_src_util,"sphere-quad")
const mpb_matrices_lib = joinpath(mpb_src_matrices,"matrices")

#

function sphrqd_dist2(x1,y1,z1,x2,y2,z2)::Float64
    return ccall((:dist2,mpb_sphere_quad),Cdouble,(Float64,Float64,Float64,Float64,Float64,Float64),x1,y1,z1,x2,y2,z2)
end

function sphrqd_dist2(v1,v2)::Float64
    return ccall((:dist2,mpb_sphere_quad),Cdouble,(Float64,Float64,Float64,Float64,Float64,Float64),v1[1],v1[2],v1[3],v2[1],v2[2],v2[3])
end

get_NQUAD2() = unsafe_load(cglobal((:NQUAD2,mpb_sphere_quad)))

x1,y1,z1,x2,y2,z2 = rand(Float64,6)

d1 = dist21(x1,y1,z1,x2,y2,z2)

dist21(rand(3),rand(3))


"""
typedef struct {
    int N, localN, Nstart, allocN;
    int c;
    int n, p, alloc_p;
    scalar *data;
} evectmatrix;

evectmatrix create_evectmatrix(int N, int c, int p,
			       int localN, int Nstart, int allocN)
{
     evectmatrix X;
 
     CHECK(localN <= N && allocN >= localN && Nstart < N,
	   "invalid N arguments");
    
     X.N = N;
     X.localN = localN;
     X.Nstart = Nstart;
     X.allocN = allocN;
     X.c = c;
     
     X.n = localN * c;
     X.alloc_p = X.p = p;
     
     if (allocN > 0) {
	  CHK_MALLOC(X.data, scalar, allocN * c * p);
     }
     else
	  X.data = NULL;

     return X;
}

void destroy_evectmatrix(evectmatrix X)
{
     free(X.data);
}


"""

"""
typedef struct {
    int p, alloc_p;
    scalar *data;
} sqmatrix;
"""

mutable struct evectmatrix
end

function evectmatrix_alloc(N::Integer, c::Integer, p::Integer, localN::Integer, Nstart::Integer, allocN::Integer)
    output_ptr = ccall(
        (:create_evectmatrix, mpb_matrices_lib),    # name of C function and library
        Ptr{evectmatrix},                           # output type
        (Cint,Cint,Cint,Cint,Cint,Cint),            # tuple of input types
        (N,c,p,localN,Nstart,allocN),               # name of Julia variable to pass in
    )
end

typedef struct {
    int p, alloc_p;
    scalar *data;
} sqmatrix;