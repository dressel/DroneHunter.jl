######################################################################
# mcts.jl
#
# like the first one, but graph-based
######################################################################

using POMDPs
using MCTS

using FEBOL

# State type
struct TS
    x::Float64      # x-position of UAV
    y::Float64      # y-position of UAV
    h::Float64      # heading of UAV

    b::ParticleCollection{TargetTuple}

    r::Float64
    kids::Matrix{Int}
end
TS(x, b) = TS(x[1], x[2], x[3], b, 0.0, zeros(Int,4,2))
TS(x, b, r) = TS(x[1], x[2], x[3], b, r, zeros(Int,4,2))
TS(x, b, r, arr) = TS(x[1], x[2], x[3], b, r, arr)

const TA = NTuple{3,Float64}


struct Graph
    belief_nodes::Vector{TS}
end

mutable struct BMDP{S,TC,TB} <: MDP{TS, Int}
    sensor::S
    lambda::Float64

    action_list::Vector{TA}
    discount_factor::Float64    # does this do anything (not really)
    cache::TC
    pf::TB
    discrete_b::Matrix{Float64}
end
POMDPs.discount(::BMDP) = 1.0
function BMDP(lambda::Float64, tc, f)
    action_list = make_action_list(5, 8, [-20, 0, 20])
    discount_factor = 0.99

    return BMDP(FOV(), lambda, action_list, discount_factor, tc, f, zeros(50,50))
end

POMDPs.actions(bmdp::BMDP) = 1:length(bmdp.action_list)

function POMDPs.generate_sr(bmdp::BMDP, s::TS, a::Int, rng::MersenneTwister)
    na = length(bmdp.action_list)

    # TODO: handle heading and bounds correctly
    # update UAV state with action
    p = new_pose( (s.x,s.y,s.h), bmdp.action_list[a] )

    # sample a state from the belief
    #random_particle = sample_state(s.b)
    random_particle = rand(Base.GLOBAL_RNG, s.b)

    # generate a possible observation
    o = observe(random_particle, bmdp.sensor, p)

    # now check if this observation has an associated belief
    if s.kids[a,o+1] == 0
        # create new belief
        bp = update_b(bmdp.pf, s.b, p, o)

        # compute reward
        penalty = bmdp.lambda * FEBOL.nmac_penalty(bp, p, 15., bmdp.pf.n)
        r = penalty + cheap_entropy(bp, bmdp.discrete_b, 200.0)
        #r = penalty + cheap_entropy(bp, 200.0, 50)

        sp = TS(p, bp, -r, zeros(Int,na,2))

        # put sp into cache and update kids list
        push!(bmdp.cache, sp)
        s.kids[a, o+1] = length(bmdp.cache)
    else
        sp = bmdp.cache[s.kids[a, o+1]]
    end

    return sp, sp.r
end

# ok, let's create a FEBOL policy
struct MCTSPolicy{BM,P} <: FEBOL.Policy
    action_list::Vector{TA}
    bmdp::BM
    policy::P
    np::Int     # number of particles in reduced pf
end
function MCTSPolicy(f::PF;
                    lambda::Float64=0.0,
                    n_iterations::Int=50,
                    depth::Int=5
                   )
    action_list = make_action_list(5, 8, [-20, 0, 20])
    na = length(action_list)
    s = TS( (0.0,0.0,0.0), f._b, 0.0, zeros(Int, na, 2))
    bmdp = BMDP(lambda, [s], f)

    gr = MothRollout(action_list)
    solver = MCTSSolver(n_iterations=n_iterations,
                        depth=depth,
                        exploration_constant=5.0)
                        #estimate_value=RolloutEstimator(gr)
                       #)
    policy = solve(solver, bmdp)

    return MCTSPolicy(bmdp.action_list, bmdp, policy, f.n)
end
function FEBOL.action(m::SearchDomain, x::Vehicle, o, f::AbstractFilter, p::MCTSPolicy)
    npf = sample_states(f, p.np)

    # create cache and first node
    na = length(p.action_list)
    s = TS( (x.x,x.y,x.heading), npf, 0.0, zeros(Int, na, 2))
    p.bmdp.cache = [s]

    # determine the action
    a_idx = POMDPs.action(p.policy, s)

    return p.action_list[a_idx]
end

# just step to it
struct MothRollout <: POMDPs.Policy
    action_list::Vector{TA}
end
function POMDPs.action(p::MothRollout, s::TS)
    x = (s.x, s.y, s.h)
    dmin = Inf
    amin = 1
    tx = centroid(s.b)
    ai = 1
    for a in p.action_list
        xp = new_pose(x, a)
        d = FEBOL.get_distance2(x, tx)
        if dmin < d
            dmin = d
            amin = ai
        end
        ai += 1
    end

    return amin
end

#end #module
