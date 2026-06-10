using OptiMode, Test, FFTW

Δx  =   3.0
Δy  =   4.0
Δz  =   5.0
Nx  =   64
Ny  =   128
Nz  =   256

gr2 =   Grid(Δx,Δy,Nx,Ny)
gr3 =   Grid(Δx,Δy,Δz,Nx,Ny,Nz)

# Check that 2D and 3D grids are equal where applicable
@test xc(gr2) ≈ xc(gr3)
@test yc(gr2) ≈ yc(gr3)
# @test zc(gr2) ≈ zc(gr3)


@test xc(gr2)[1:end-1]  ≈   ( OptiMode.x(gr2) .- (δx(gr2)/2.0,) )
@test xc(gr2)[2:end]    ≈   ( OptiMode.x(gr2) .+ (δx(gr2)/2.0,) )
@test yc(gr2)[1:end-1]  ≈   ( OptiMode.y(gr2) .- (δy(gr2)/2.0,) )
@test yc(gr2)[2:end]    ≈   ( OptiMode.y(gr2) .+ (δy(gr2)/2.0,) )

@test xc(gr3)[1:end-1]  ≈   ( OptiMode.x(gr3) .- (δx(gr3)/2.0,) )
@test xc(gr3)[2:end]    ≈   ( OptiMode.x(gr3) .+ (δx(gr3)/2.0,) )
@test yc(gr3)[1:end-1]  ≈   ( OptiMode.y(gr3) .- (δy(gr3)/2.0,) )
@test yc(gr3)[2:end]    ≈   ( OptiMode.y(gr3) .+ (δy(gr3)/2.0,) )
@test zc(gr3)[1:end-1]  ≈   ( OptiMode.z(gr3) .- (δz(gr3)/2.0,) )
@test zc(gr3)[2:end]    ≈   ( OptiMode.z(gr3) .+ (δz(gr3)/2.0,) )

# Test that `corners(g::Grid)` produces pixel/voxel corners which average to the corresponding grid points
@test sum.(corners(gr2))./4 ≈ vec(x⃗(gr2))
@test sum.(corners(gr3))./8 ≈ vec(x⃗(gr3))

@test isapprox(fftfreq(gr.Nx,gr.Nx/gr.Δx),my_fftfreq(gr.Nx,gr.Nx/gr.Δx))
@test isapprox(fftfreq((gr.Nx+1),(gr.Nx+1)/gr.Δx),my_fftfreq((gr.Nx+1),(gr.Nx+1)/gr.Δx))

ff_sum2D(Dx_Dy) = sum(sum(g⃗(Grid(Dx_Dy[1],Dx_Dy[2],256,128))))
@test Zygote.gradient(ff_sum2D,[6.0,4.0])[1] ≈ ForwardDiff.gradient(ff_sum2D,[6.0,4.0])
