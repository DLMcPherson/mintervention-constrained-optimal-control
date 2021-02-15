import numpy as np
import dlqr
import objective_functions as of
import quadratic_objectives as qof

# TODO: break into smaller functions

def iterative_LQR(seed_states,seed_control,timestep_length, dynamics,runcost,terminal_cost = 0, step_size=1,ITER_NUM=10,starting_regularizer=1,alwaysCheck=True,neverRetreat=False):

    # Copy the seed trajectory to initialize our trajectory optimization
    states = np.copy(seed_states)
    z0 = seed_states[:,0]
    control = np.copy(seed_control)
    N = seed_control.shape[1]

    # Handle default terminal cost
    if(terminal_cost == 0):
        # default to zero terminal cost
        terminal_cost = of.TerminalCost(qof.QuadraticStateCost(dlqr.QuadraticForm(dynamics.dimZ,np.zeros((dynamics.dimZ,dynamics.dimZ)),np.zeros((dynamics.dimZ,1)),0 ) ))

    # Compute the cost of the initial trajectory
    cost = 0
    for i in np.arange(0,N):
        cost = cost + timestep_length*runcost.f(states[:,i],control[:,i],i)
    cost = cost + timestep_length*terminal_cost.f(states[:,N])

    print("Initialization's cost is ",cost[0,0])
    # ==================== iterative Linear Quadratic Regulator Optimization ================== #
    lamb = np.copy(starting_regularizer)
    for iLQR_iterations in np.arange(1,ITER_NUM):
        # Copy previous loop's trajectory, control trace, and cost
        oldControl = np.copy(control)
        oldStates = np.copy(states)
        oldCost = np.copy(cost)

        KN, kN = solveLocalLQR(oldStates,oldControl, dynamics,runcost,terminal_cost, timestep_length,N, lamb)

        # Simulate the result of this time-varying LQR control policy on the true non-linear dynamics
        for armijo_step in np.append(1/np.power(2.0,np.arange(-2,1)),0): # test out a variety of feedforward gains
            cost, states, control = assessControlUpdate(oldControl,oldStates,KN,kN, runcost,terminal_cost, z0,dynamics,N,timestep_length, armijo_step,step_size)
            #print(70000,armijo_step,cost)
            # Stop Armijo backtracking when cost improves over previous
            if(oldCost - cost > 0):
                break

        print(iLQR_iterations,cost[0,0])
        # Assess the efficacy of this update loop after initial burn-in (optional)
        # (This code block is widely heuristic, so feel free to fiddle with it as you please)
        if( (iLQR_iterations > ITER_NUM*0.1 or alwaysCheck) and neverRetreat==False):
            # Check if this update improved upon the objective keep the result
            if(cost < oldCost):
                lamb = lamb/2.0 # If this LQR approximation step worked, trust the quadratic approximation more
                # If the percent improvement on the objective is minor, then the future improvements will likely be minor
                # (motivated by magnitude of graident being upper bounded by objective distance from minimum for strongly convex objectives)
                if(np.abs(cost-oldCost)/cost < 0.001):
                    break
            # otherwise this update was worse
            else:
                print("Worse by ",100*np.abs(cost[0,0]-oldCost[0,0])/cost[0,0],"%")
                # if the update was worse by more than 4% reset to prior art
                # otherwise allow this minor hill-climb, we may be reaching a better minimum!
                if(np.abs(cost-oldCost)/cost > 0.04):
                    # reset to prior art
                    states = oldStates
                    control = oldControl
                    cost = oldCost
                    print("resetting")
                    # trust quadratic approximation less (update method approaches gradient-step rather than Newton-Raphson)
                    lamb *= 4
                    if lamb > 90000:
                        print("Eigenvalue regularization exceeded; breaking")
                        break


    # Return the control trace
    return(states, control)

def iterative_LegibleLQR(seed_states,seed_control,timestep_length, dynamics,runcost,terminal_cost, dynamicsLarger,runcostLarger,terminal_costLarger, step_size=1,ITER_NUM=10,starting_regularizer=1,alwaysCheck=True,neverRetreat=False):

    # Copy the seed trajectory to initialize our trajectory optimization
    states = np.copy(seed_states)
    z0 = seed_states[:,0]
    control = np.copy(seed_control)
    N = seed_control.shape[1]

    # Handle default terminal cost
    if(terminal_cost == 0):
        # default to zero terminal cost
        terminal_cost = of.TerminalCost(qof.QuadraticStateCost(dlqr.QuadraticForm(dynamics.dimZ,np.zeros((dynamics.dimZ,dynamics.dimZ)),np.zeros((dynamics.dimZ,1)),0 ) ))

    # Compute the cost of the initial trajectory
    ContainedCost = 0
    LargerCost = 0
    for i in np.arange(0,N):
        ContainedCost = ContainedCost + timestep_length*runcost.f(states[:,i],control[:,i],i)
        LargerCost = LargerCost + timestep_length*runcostLarger.f(states[:,i],control[:,i],i)
    ContainedCost = ContainedCost + timestep_length*terminal_cost.f(states[:,N])
    LargerCost = LargerCost + timestep_length*terminal_costLarger.f(states[:,N])
    cost = ContainedCost - LargerCost
    #cost = np.exp(ContainedCost) / (np.exp(ContainedCost) + np.exp(LargerCost)) # treating the partitions as equal (patently false)
    cost = ContainedCost

    print("Initialization's cost is ",cost[0,0])
    # ==================== iterative Linear Quadratic Regulator Optimization ================== #
    lamb = np.copy(starting_regularizer)
    for iLQR_iterations in np.arange(1,ITER_NUM):
        # Copy previous loop's trajectory, control trace, and cost
        oldControl = np.copy(control)
        oldStates = np.copy(states)
        oldCost = np.copy(cost)

        KN, kN = solveLocalLQR(oldStates,oldControl, dynamics,runcost,terminal_cost, timestep_length,N, lamb)
        KNLarger, kNLarger = solveLocalLQR(oldStates,oldControl, dynamicsLarger,runcostLarger,terminal_costLarger, timestep_length,N, lamb)

        # Simulate the result of this time-varying LQR control policy on the true non-linear dynamics
        for armijo_step in np.append(1/np.power(2.0,np.arange(-5,5)),0): # test out a variety of feedforward gains
            ContainedCost, states, control = assessControlUpdate(oldControl,oldStates,KN-KNLarger,kN-kNLarger, runcost,terminal_cost, z0,dynamics,N,timestep_length, armijo_step,step_size)

            # assess the cost of the containing condition as well
            LargerCost = 0
            for i in np.arange(0,N):
                # Tally the running cost
                LargerCost = LargerCost + timestep_length*runcostLarger.f(states[:,i],control[:,i],i)
            # Add the terminal cost
            LargerCost = LargerCost + timestep_length*terminal_costLarger.f(states[:,N])

            # estimate the legibility equivalent objective
            cost = ContainedCost - LargerCost
            #cost = np.exp(ContainedCost) / (np.exp(ContainedCost) + np.exp(LargerCost)) # treating the partitions as equal (patently false)
            cost = ContainedCost

            # Stop Armijo backtracking when cost improves over previous
            if(oldCost - cost > -10):
                break

        print(iLQR_iterations,cost[0,0])

    # Return the control trace
    return(states,control)

def solveLocalLQR(oldStates,oldControl, dynamics,runcost,terminal_cost, timestep_length,N, lamb):
    # Linearize the dynamics around the previous trajectory
        # Initialize linearization storage arrays
    AN = np.zeros((dynamics.dimZ,dynamics.dimZ,N)) # initialize time-array of linear dynamics matrices A_t (for t = [1,N])
    BN = np.zeros((dynamics.dimZ,dynamics.dimU,N)) # initialize time-array of linear control matrices A_t (for t = [1,N])
        # Linearize at each point in time
    for i in np.arange(0,N):
        # Take first order approximation of continuous time dynamics:
        #$$    f(x,u,t) ~= \nabla_x f(x^{old},u^{old},t) (x - x^{old}) + \nabla_u f(x^{old},u^{old},t) (u - u^{old})    $$
        # and also approximate in discrete-time with Euler integration approximation:
        #$$    x_{t+1} = x_{t} + \Delta t * f(x,u,t)    $$
        AN[:,:,i] = np.eye(dynamics.dimZ) + timestep_length*dynamics.deriv_x(oldStates[:,i],oldControl[:,i])
        BN[:,:,i] = timestep_length*dynamics.deriv_u(oldStates[:,i],oldControl[:,i])
    # Quadraticize the cost functional around the current trajectory and control trace:
    #$$    \int J(x,u,t) dt ~= \sum \Delta t (J(x^{old},u^{old},t) + \nabla_x J(x^{old},u^{old},t) (x - x^{old}) + (x - x^{old})^T \nabla^2_{xx} J(x^{old},u^{old},t) (x - x^{old})
    #                                   + \nabla_u J(x^{old},u^{old},t) (u - u^{old}) + (u - u^{old})^T \nabla^2_{uu} J(x^{old},u^{old},t) (u - u^{old}) ) $$
    QN = [dlqr.QuadraticForm(dynamics.dimZ,
        timestep_length*0.5*runcost.deriv_statestate(oldStates[:,i],oldControl[:,i],i), # Matrix coefficient for quadratic term
        timestep_length*runcost.deriv_state(oldStates[:,i],oldControl[:,i],i), # Vector coefficient for linear term
        timestep_length*runcost.f(oldStates[:,i],oldControl[:,i],i) ) for i in np.arange(0,N)] # Scalar coefficient for affine term
    RN = [dlqr.QuadraticForm(dynamics.dimU,
        timestep_length*0.5*runcost.deriv_controlcontrol(oldStates[:,i],oldControl[:,i],i), # Matrix coefficient for quadratic term
        timestep_length*runcost.deriv_control(oldStates[:,i],oldControl[:,i],i), # Vector coefficient for linear term
        0 ) for i in np.arange(0,N)] # Scalar coefficient for affine term
    # Quadraticize terminal state cost around current trajectories' end state
    quadraticized_terminal_cost = dlqr.QuadraticForm(dynamics.dimZ,
        timestep_length*0.5*terminal_cost.deriv_statestate(oldStates[:,N]),
        timestep_length*terminal_cost.deriv_state(oldStates[:,N]),
        timestep_length*terminal_cost.f(oldStates[:,N]) )

    # Find the LQR optimal response to this linearization and quadraticization
    KN, kN = solveLQR(AN,BN,QN,RN,quadraticized_terminal_cost,dynamics.dimZ,dynamics.dimU,N, lamb)

    return(KN, kN)

def assessControlUpdate(oldControl,oldStates,KN,kN, runcost,terminal_cost, z0,dynamics,N,timestep_length, armijo_step,step_size):
    # Clear out the trajectory, control trace, and cost
    cost = 0
    control = np.zeros((dynamics.dimU,N))
    states = np.zeros((dynamics.dimZ,N+1))
    # Initialize state
    states[:,0] = z0
    # Iterate forward through time
    for i in np.arange(0,N):
        # Add the LQR control perturbation on top of the setpoint control trace used in the Taylor expansion
        control[:,i] = oldControl[:,i] + step_size*(np.dot(KN[:,:,i],(states[:,i]-oldStates[:,i])) + armijo_step*kN[:,0,i])
        # Euler integrate the state dynamics in response to the control
        states[:,i+1] = states[:,i] + timestep_length * dynamics.f(states[:,i],control[:,i])
        # Wrap the angle state # TODO: Generalize this state periodicitiy for other dynamics
        '''
        if(states[2,i+1] > np.pi):
            states[2,i+1] = states[2,i+1] - 2*np.pi
        if(states[2,i+1] < -np.pi):
            states[2,i+1] = states[2,i+1] + 2*np.pi
        '''
        # Tally the running cost
        cost = cost + timestep_length*runcost.f(states[:,i],control[:,i],i)
    # Add the terminal cost
    cost = cost + timestep_length*terminal_cost.f(states[:,N])
    return(cost, states, control)

def solveLQR(AN,BN,state_costs,control_costs,terminal_state_cost, num_of_state_dimensions,num_of_control_dimensions,time_horizon_length, damping_parameter):
    # Initialize storage arrays
    PN = [dlqr.QuadraticForm(num_of_state_dimensions,np.zeros((num_of_state_dimensions,num_of_state_dimensions)),np.zeros((num_of_state_dimensions,1)),0) for n in np.arange(0,time_horizon_length+1)]
    PN[time_horizon_length] = terminal_state_cost
    KN = np.zeros((num_of_control_dimensions,num_of_state_dimensions,time_horizon_length)) # Allocate the feedback gain schedule array
    kN = np.zeros((num_of_control_dimensions,1,time_horizon_length)) # Allocate the feedforward schedule array
    # Iterate backwards through time, starting at the desired outcome and stepping backwards with Bellman updates
    for i in ((time_horizon_length-1)-np.arange(0,time_horizon_length)):
        KN[:,:,i], kN[:,:,i], PN[i], eigvals = dlqr.onestep_dlqr_affine(AN[:,:,i],BN[:,:,i],state_costs[i],control_costs[i],PN[i+1],damping_parameter)
        #print(PN[i])
    return(KN, kN)
