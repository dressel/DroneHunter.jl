######################################################################
# batch.jl
# writing my own batch sim code
######################################################################

using FEBOL

export my_batch
#export my_parsim

# for running batches of simulations on a vector of sim units
function my_batch(m::SearchDomain, suv, n_sims::Int)

    theta_start = m.theta
    x0 = suv[1].x.x
    y0 = suv[1].x.y
    h0 = suv[1].x.heading

    costs = zeros(n_sims, length(suv))
    nmacs = zeros(n_sims, length(suv))
    d2_errors = zeros(n_sims, length(suv))
    for sim_ind = 1:n_sims

        # reset jammer to random location

        print("Simulation ", sim_ind, ": ")
        for (su_ind, su) in enumerate(suv)
            m.theta = theta_start
            print(su_ind, ",")

            # reset the filter, vehicle, and policy
            #reset!(m, su)
            reset!(su.f)

            # reset vehicle
            reset!(m, su.x)
            su.x.x = 100.0
            su.x.y = 10.0
            su.x.heading = 0.0
            #su.x.x = x0
            #su.x.y = y0
            #su.x.heading = h0

            # does policy need to be reset?


            c1, c2, c3 = my_simulate(m, su)
            costs[sim_ind, su_ind] = c1
            nmacs[sim_ind, su_ind] = c2
            d2_errors[sim_ind, su_ind] = c3
        end
        println("complete.")
    end
    return costs, nmacs, d2_errors
end

function my_simulate(m::SearchDomain, su::SimUnit)

    # Don't reset the filter, vehicle, and policy
    # I think I assume the SimUnit comes in clean and ready to go

    # What was the cost to getting this first observation?
    #cost_sum = get_cost(su, m)
	cost_sum = 0.0			# really just sum of entropy
    nmac_cost = 0.0
    error_cost = 0.0
	total_cost = 0.0		# sum of entropy and collision costs, including lambda

    # before doing anything else, we observe
    #  and update filter once
    o = observe(m, su.x)
    update!(su, o)

    # This was our first step; steps count number of observations
    step_count = 1

    while !is_complete(su.f, su.tc, step_count)
        # act
        a = FEBOL.action(m, su, o)
        act!(m, su.x, a)

        move_target!(m)

        # is there an nmac?
        dx = su.x.x - m.theta[1]
        dy = su.x.y - m.theta[2]
        if sqrt(dx*dx + dy*dy) < 15.0
            nmac_cost += 1.0
			total_cost += su.p.bmdp.lambda
        end

        # observe and update
        o = observe(m, su.x)
        update!(su, o)

        # get cost and update step count
		ec = get_cost(su, m, a)
        cost_sum += ec
		total_cost += ec
        step_count += 1

        if step_count > 20
            c = centroid(su.f)
            dx = c[1] - m.theta[1]
            dy = c[2] - m.theta[2]
            error_cost += sqrt(dx*dx + dy*dy)
        end
    end

    return cost_sum, nmac_cost, error_cost, total_cost
end
# like simulation, but special for parallel simulations
# main change is deep-copying of arguments so as to not disturb them
function psim(idx::Int, m::SearchDomain, vsu::Vector{SimUnit}, theta_start::NTuple{4, Float64})

    # copy these so they don't get messed up by other cores
    m = deepcopy(m)
    vsu = deepcopy(vsu)

    # change the target location
    theta!(m)

    costs = zeros(length(vsu))
    nmacs = zeros(length(vsu))
    d2s = zeros(length(vsu))
    totals = zeros(length(vsu))

    print("Simulation ", idx, ": ")
    for (su_ind, su) in enumerate(vsu)
        m.theta = theta_start
        print(su_ind, ",")

        # reset the filter, vehicle, and policy
        reset!(m, su)
        reset!(su.f)

        # reset vehicle
        reset!(m, su.x)
        su.x.x = 100.0
        su.x.y = 10.0
        su.x.heading = 0.0

        c1, c2, c3, c4 = my_simulate(m, su)
        costs[su_ind] = c1
        nmacs[su_ind] = c2
        d2s[su_ind] = c3
        totals[su_ind] = c4
    end
    println("complete.")
    return costs, nmacs, d2s, totals
end

function my_parsim(m::SearchDomain, vsu::Vector{SimUnit}, n_sims::Int)
    np = nprocs()       # number of processes available
    i = 1
    nextidx() = (idx=i; i+=1; idx)
    #costs = zeros(n_sims, length(vsu))

    costs = zeros(n_sims, length(vsu))
    nmacs = zeros(n_sims, length(vsu))
    d2_errors = zeros(n_sims, length(vsu))
    totals = zeros(n_sims, length(vsu))

    theta_start = m.theta
    @sync begin
        for p=1:np
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        idx > n_sims && break
                        c1,c2,c3,c4=remotecall_fetch(psim, p, idx, m, vsu, theta_start)
                        costs[idx,:] = c1
                        nmacs[idx,:] = c2
                        d2_errors[idx,:] = c3
                        totals[idx,:] = c4
                    end
                end
            end
        end
    end
    return costs, nmacs, d2_errors, totals
end
function my_parsim(m::SearchDomain, su::SimUnit, n_sims::Int)
    return vec( my_parsim(m, [su], n_sims) )
end

