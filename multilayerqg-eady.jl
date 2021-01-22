# # Eady model of Baroclinic Instability
#
# A simulation of the growth of barolinic instability in the Eady n-layer model
# when we impose a vertical mean flow shear as a difference $\Delta U$ in the
# imposed, domain-averaged, zonal flow at each layer.

using FourierFlows, Plots, Printf

using FFTW: rfft, irfft
using Statistics: mean

import GeophysicalFlows.MultiLayerQG
import GeophysicalFlows.MultiLayerQG: energies

# ## Numerical parameters and time-stepping parameters
nx = 256                  # 2D resolution = nx^2
ny = nx

stepper = "FilteredRK4"   # timestepper
dt = 2e-3      # timestep
nsteps = 20000            # total number of time-steps
nsubs  = 100              # number of time-steps for plotting (nsteps must be multiple of nsubs)

# ## Physical parameters
Lx = 2π         # domain size
 μ = 1e-2       # bottom drag
 β = 0.1          # the y-gradient of planetary PV
 g = 1.0
f₀ = 2.0
 
# Vertical grid
nlayers = 5     # number of layers

total_depth = 1
Δh = total_depth / nlayers
z = [ (i - 1/2) * Δh for i = nlayers:-1:1 ] .- total_depth
H = [ Δh for i = 1:nlayers ]

# Density stratification
ρ₀ = 1
N² = 5

ρ_resting(z) = ρ₀ * (1 - N² * z / g)
ρ = ρ_resting.(z)

@info @sprintf("The largest Rossby radius of deformation is %.2f",
               sqrt(N²) * total_depth / f₀) 

@info @sprintf("The smallest Rossby radius of deformation is %.2f",
               sqrt(N²) * Δh / f₀) 


# Background shear
ΔU = 1 / (nlayers - 1)
U = [i * ΔU for i=nlayers-1:-1:0]

twod_grid = TwoDGrid(nx, Lx)

X, Y = gridpoints(twod_grid)
σx, σy = 0.7, 0.5
eta = f₀ / sum(H) * Δh * @. exp(-X^2/σx^2 -Y^2/σy^2)

# ## Problem setup
# 
# We initialize a `Problem` by providing a set of keyword arguments,

prob = MultiLayerQG.Problem(nlayers,
                             nx = nx,
                             Lx = Lx,
                             f₀ = f₀,
                              g = g,
                              H = H,
                              ρ = ρ,
                              U = U,
                             dt = dt,
                        stepper = stepper,
                            eta = eta,
                              μ = μ, 
                              β = β
                           )

# and define some shortcuts.
sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = grid.x, grid.y

# ## Setting initial conditions

# Our initial condition is some small amplitude random noise. We smooth our initial
# condidtion using the `timestepper`'s high-wavenumber `filter`.

qᵢ  = randn((nx, ny, nlayers))
for j in 1:nlayers
  qᵢ[:, :, j] = qᵢ[:, :, j] .- mean(qᵢ[:, :, j])
end
qhᵢ = prob.timestepper.filter .* rfft(qᵢ, (1, 2)) # only apply rfft in dims=1, 2
qᵢ  = irfft(qhᵢ, grid.nx, (1, 2)) # only apply irfft in dims=1, 2

MultiLayerQG.set_q!(prob, qᵢ)


KEᵢ, PEᵢ = MultiLayerQG.energies(prob)
KEᵢ

ψhᵢ = similar(qhᵢ)

MultiLayerQG.streamfunctionfrompv!(ψhᵢ, qhᵢ, prob.params, prob.grid)

for j in 1:nlayers
  ψhᵢ[:, :, j] = sqrt(5e-8 / KEᵢ[j]) * ψhᵢ[:, :, j]
end
ψᵢ  = irfft(ψhᵢ, grid.nx, (1, 2))

MultiLayerQG.set_ψ!(prob, ψᵢ)
nothing # hide

# ## Diagnostics

# Create Diagnostics -- `energies` function is imported at the top.
E = Diagnostic(energies, prob; nsteps=nsteps)
diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.

nothing # hide

# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_eady"
plotname = "snapshots"
filename = joinpath(filepath, "eady.jld2")
nothing # hide

# Do some basic file management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# And then create Output
get_sol(prob) = prob.sol # extracts the Fourier-transformed solution

out = Output(prob, filename, (:sol, get_sol))
nothing # hide

# ## Visualizing the simulation

# We define a function that plots the potential vorticity field and the evolution 
# of energy and enstrophy.

function plot_output(prob)
  q = prob.vars.q
  x, y = prob.grid.x, prob.grid.y
  Lx, Ly = prob.grid.Lx, prob.grid.Ly
  
  layout = @layout Plots.grid(2, Int(round(nlayers/2)+1)+1)
  
  p = plot(layout=layout, size=(1000, 600), dpi=180)
  
  for m in 1:nlayers

    heatmap!(p[m], x, y, q[:, :, m],
         aspectratio = 1,
              legend = :none,
                   c = :deep,
               xlims = (-Lx/2, Lx/2),
               ylims = (-Ly/2, Ly/2),
               title = "q layer "*string(m),
              xticks = :none,
              yticks = :none,
            colorbar = false,
          framestyle = :box

         )
  end
  
  plot!(p[nlayers+1], nlayers,
             # label = ["KE1" "KE2"],
            legend = false,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.05*μ*nsteps*dt, 1.01*μ*nsteps*dt),
             ylims = (1e-8, 1e2),
            yscale = :log10,
            yticks = 10.0.^(-8:2:2),
            xlabel = "μt",
             title = "KE at layers")
          
  plot!(p[nlayers+2], nlayers-1,
             # label = "PE",
            legend = false,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.05*μ*nsteps*dt, 1.01*μ*nsteps*dt),
             ylims = (1e-9, 1e1),
            yscale = :log10,
            yticks = 10.0.^(-8:2:2),
            xlabel = "μt",
             title = "PE at interfaces")

  return p
end

nothing # hide

# ## Time-stepping the `Problem` forward

# Finally, we time-step the `Problem` forward in time.

p = plot_output(prob)

startwalltime = time()

anim = @animate for j = 0:Int(nsteps / nsubs)
  
    cfl = clock.dt * maximum([maximum(abs, vars.u) / grid.dx, maximum(abs, vars.v) / grid.dy])
    
    log = @sprintf("step: %04d, dt: %.3f, t: %0.2f, cfl: %.2f, walltime: %.2f min", 
                    clock.step, clock.dt, clock.t, cfl, (time()-startwalltime)/60)

    log_KE = @sprintf("KE₁: %.3e     KE₂: %.3e     KE₃: %.3e     KE₄: %.3e     KE₅: %.3e",
                  E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][1][3], E.data[E.i][1][4], E.data[E.i][1][5])

    log_PE = @sprintf("      PE₁.₅: %.3e   PE₂.₅: %.3e   PE₃.₅: %.3e   PE₄.₅: %.3e",
                  E.data[E.i][2][1], E.data[E.i][2][2], E.data[E.i][2][3], E.data[E.i][2][4])
                  
    # log_KE = @sprintf("KE₁: %.3e     KE₂: %.3e     KE₃: %.3e",
    #               E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][1][3])
    # 
    # log_PE = @sprintf("      PE₁.₅: %.3e   PE₂.₅: %.3e",
    #               E.data[E.i][2][1], E.data[E.i][2][2])
    # 

    if j % (500 / nsubs) == 0
      println(log)
      println(log_KE)
      println(log_PE)
    end
    
    for m in 1:nlayers
      p[m][1][:z] = @. vars.q[:, :, m] + params.β * grid.y'
      if m==nlayers
        p[m][1][:z] = @. vars.q[:, :, m] + params.β * grid.y' + params.eta
      end
      push!(p[nlayers+1][m], μ*E.t[E.i], E.data[E.i][1][m])
    end
    
    for m in 1:nlayers-1
      push!(p[nlayers+2][m], μ*E.t[E.i], E.data[E.i][2][m])
    end

    stepforward!(prob, diags, nsubs)

    MultiLayerQG.updatevars!(prob)

end

mp4(anim, "multilayerqg_eady.mp4", fps=18)

# ## Save

#=
# Finally save the last snapshot.
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), clock.step)
savefig(savename)
=#
