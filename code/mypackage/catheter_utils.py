# Sympy libraries to symbolically create system models
from sympy import symbols
from sympy.physics import mechanics
from sympy.physics.mechanics import dynamicsymbols
import sympy.physics.mechanics as me
from sympy.physics.mechanics import inertia
from sympy import Dummy, lambdify
from sympy import Matrix

# Scipy libraries to optimize and integrate
from scipy import integrate
from scipy.integrate import odeint
from scipy.optimize import minimize

# Numpy library to perform computational heavy lifting
import numpy as np

def derive_sys(n,p):
    """
    Derive the equations of motion using Kane's method
    
    Inputs:
        n - number of discrete elements to use in the model
        p - packed parameters to create symbolic material and physical equations
    
    Outputs:
        Symbolic, nolinear equations of motion for the discretized catheter model
    
    """
    #-------------------------------------------------
    # Step 1: construct the catheter model
    
    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass) 
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))
    
    # Torques applied to each element due to external loads
    Torque = dynamicsymbols('tau:{0}'.format(n))
    
    # Force applied at the end of the catheter by the user
    F_in = dynamicsymbols('F:{0}'.format(1))

    # Unpack the system values
    M, L, E, I = p
    
    # Structural damping 
    damp = 0.05

    # Assuming that the lengths and masses are evenly distributed 
    # (that is, the sytem is homogeneous), let's evenly divide the
    # Lengths and masses along each discrete member
    lengths = np.concatenate([np.broadcast_to(L / n,n)])    
    masses = np.concatenate([np.broadcast_to(M / n,n)])

    # time symbol
    t = symbols('t')
    
    # The stiffness of the internal springs simulating material stiffness
    stiffness = E * I
    
    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    P.set_vel(A,0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []
    
    # Create a rotated reference frame for the first rigid link
    Ar = A.orientnew('A' + str(0), 'axis', [q[0],A.z])

    # Create a point at the center of gravity of the first link
    Gr = P.locatenew('G' + str(0),(lengths[0] / 2) * Ar.x)
    Gr.v2pt_theory(P,A,Ar)
    
    # Create a point at the end of the link
    Pr = P.locatenew('P' + str(0), lengths[0] * Ar.x)
    Pr.v2pt_theory(P, A, Ar)   

    # Create the inertia for the first rigid link
    Inertia_r = inertia(Ar,0,0,masses[0] * lengths[0]**2 / 12)

    # Create a new particle of mass m[i] at this point
    Par = mechanics.RigidBody('Pa' + str(0), Gr, Ar, masses[0], (Inertia_r,Gr))
    particles.append(Par)
    
    # Add an internal spring based on Euler-Bernoulli Beam theory
    forces.append((Ar, -stiffness * (q[0]) / (lengths[0]) * Ar.z))
    
    # Add a damping term
    forces.append((Ar, (-damp * u[0]) * Ar.z))
    
    # Add a new ODE term
    kinetic_odes.append(q[0].diff(t) - u[0])
    
    P = Pr
    
    for i in range(1,n):
        
        # Create a reference frame following the i^th link
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i],Ar.z])
        Ai.set_ang_vel(A, u[i] * Ai.z)
        
        # Set the center of gravity for this link
        Gi = P.locatenew('G' + str(i),lengths[i] / 2 * Ai.x)
        Gi.v2pt_theory(P,A,Ai)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), lengths[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)
        
        # Set the inertia for this link
        Inertia_i = inertia(Ai,0,0,masses[i] * lengths[i]**2 / 12)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.RigidBody('Pa' + str(i), Gi, Ai, masses[i], (Inertia_i,Gi))
        particles.append(Pai)
        
        # The external torques influence neighboring links
        if i + 1 < n:
            next_torque = 0
            for j in range(i,n):
                next_torque += Torque[j]
        else:
            next_torque = 0.
            
        forces.append((Ai,(Torque[i] + next_torque) * Ai.z))
        
        # Add another internal spring
        forces.append((Ai, (-stiffness * (q[i] - q[i-1]) / (2 * lengths[i])) * Ai.z))
        
        # Add the damping term
        forces.append((Ai, (-damp * u[i]) * Ai.z))

        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi
    
    # Add the user-defined input at the tip of the catheter, pointing normal to the 
    # last element
    forces.append((P, F_in[0] * Ai.y))

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations( particles, forces)

    return KM, fr, fr_star, q, u, Torque, F_in, lengths, masses

def parameterize(n,kane):
    """
    Parameterize the symbolic equations of motion so that they can be integrated 
    
    Inputs:
        n - number of elements
        kane - full nonlinear equations of motion

    Outputs: 
        mm_func - mass matrix function
        fo_func - forcing matrix function
    """

    # Unpack the kanes method parameters
    KM, fr, fr_star, q, u, tau, force_in, lengths, masses = kane
    
    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u + tau + list(force_in)]
    unknown_dict = dict(zip(q + u + tau + list(force_in), unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation 
    mm_func = lambdify(unknowns, mm_sym)
    fo_func = lambdify(unknowns, fo_sym)

    return mm_func,fo_func

def linearize_system(n,kane):
    """
    Take the previously derived equations of motion and create an LTI model
    
    Inputs:
        n - number of elements
        kane - full nonlinear equations of motion
        p - packed parameters (must have values)
    
    Outputs:
        A_np - Linearized A matrix as a numpy array
        B_np - Linearized B matrix as a numpy array
    """

    # Unpack the kanes method parameters
    KM, fr, fr_star, q, u, Torque, F_in, lengths, masses = kane

    # Linearize the Kane's method equations
    linearizer = KM.to_linearizer()
    
    # Output the A, B, and Mass matrices
    Maz, A, B = linearizer.linearize()
    
    # Create an operating point around which we will linearize
    op_point = dict()

    # we will linearize about the undeflected, stationary point
    for h in range(n):
        op_point[q[h]] = 0.0
        op_point[u[h]] = 0.0
    
    # Perform substitutions to solve for the linearized matrices
    M_op = me.msubs(Maz, op_point)
    A_op = me.msubs(A, op_point)
    B_op = me.msubs(B, op_point)
    perm_mat = linearizer.perm_mat

    # Solve for the linear A and B matrices
    A_lin = perm_mat.T * M_op.LUsolve(A_op)
    B_lin = perm_mat.T * M_op.LUsolve(B_op)
    A_sol = A_lin.subs(op_point).doit()
    B_sol = B_lin.subs(op_point).doit()

    # Ensure the matrices are of the correct data type
    A_np = np.array(np.array(A_sol), np.float)
    B_np = np.array(np.array(B_sol), np.float)

    return A_np,B_np

def body_rotation(coords,angle):
    """
    Rotate a body in a two-dimensional coordinate frame
    
    Inputs:
        coords - [2xn] array of coordinates in the current frame
        angle - [1xn] array of angles about which the frame is rotated
    
    Outputs:
        x - X-coordinate in the rotated frame
        y - Y-coordinate in the rotated frame
    """
    
    # Rotation about the z-axis
    def R_z(theta):
        rotation = np.zeros([len(theta),2,2])
        rotation[:,0,0] = np.cos(theta)
        rotation[:,0,1] = -np.sin(theta)
        rotation[:,1,0] = np.sin(theta)
        rotation[:,1,1] = np.cos(theta)
        return rotation
    
    # Perform the rotation
    rotated_coords = np.matmul(R_z(angle[:]), coords)
    
    # Extract the x and y coordinates
    x = rotated_coords[:,0,0]
    y = rotated_coords[:,1,0]

    return x,y


def get_xy_coords(n,q, lengths):
    """
    Get (x, y) coordinates of the beam from generalized coordinates q
    
    Inputs:
        n - number of elements
        q - generalized coordinates
        lengths - length of each element
    
    Outputs:
        x_coords - X-coordinates of each beam element
        y_coords - Y-coordinates of each beam element
    """
    
    q = np.atleast_2d(q)
    

    zeros = np.zeros(q.shape[0])[:, None]
    x = np.hstack([zeros])
    y = np.hstack([zeros])

    for i in range(0,n):
        
        # Get the new x and y coordinates to append 
        x_app,y_app = body_rotation(np.array([[lengths[i]],[0]]),q[:,i])
        
        # Append to the matrices
        x = np.append(x,np.atleast_2d(x_app).T,axis=1)
        y = np.append(y,np.atleast_2d(y_app).T,axis=1)
    
    x_coords = np.cumsum(x,1)
    y_coords = np.cumsum(y,1) 

    return x_coords, y_coords

def get_xy_deriv(n,q, lengths):
    """
    Get (x_dot, y_dot) coordinates from generalized coordinates q

    Inputs:
        n - number of elements
        q - generalized coordinates
        lengths - length of each element
    
    Outputs:
        x_dot_coords - X_dot-coordinates of each beam element
        y_dot_coords - Y_dot-coordinates of each beam element
    
    """
    
    q = np.atleast_2d(q)

    zeros = np.zeros(q.shape[0])[:, None]
    x_dot = np.hstack([zeros])
    y_dot = np.hstack([zeros])

    # Do this for each element
    for i in range(0,n):
    
        # Get the new x_dot and y_dot coordinates to append
        x_app = -lengths[i] * np.sin(q[:,i]) * q[:,i + (n)]
        y_app = lengths[i] * np.cos(q[:,i]) * q[:,i + (n)]
        
        x_dot = np.append(x_dot,np.atleast_2d(x_app).T,axis=1)
        y_dot = np.append(y_dot,np.atleast_2d(y_app).T,axis=1)

    x_dot_coords = np.cumsum(x_dot,1)
    y_dot_coords=  np.cumsum(y_dot,1)
 
    return x_dot_coords,y_dot_coords

def force(coords,r,time):
    ''' 
    Get the force resulting from flow on a cylinder in two dimensions
    
    Inputs:
        Coords - location of the current element in the global coordinate frame
        r - displacement vector from the origin of the element
        time - current time 
    
    Outputs:
        F - [2x1] array containing X and Y forces in the reference coordinate frame
    '''
    
    # Maximum velocity of the flow field in m/s
    flow_amp = 0.5
    
    # Flow field velocity
    u_y = flow_amp * np.cos(1. * np.pi * time) + flow_amp # m/s
    
    #  Drag coefficient
    Cd = 1.2 
    
    # Mass density of blood
    rho_blood = 1060 # kg/m^3
    
    # Diameter of the catheter
    Diam = 2.667e-3 # m
    
    # Length projected into the X-axis
    projected_length = r[0]
    
    # Drag force
    F_dy = 0.5 * u_y * rho_blood * Cd * Diam * projected_length
    
    # Ignoring forces in the x-direction for now
    f_x = 0.

    return np.array([[f_x],[F_dy]])

def get_forces(n,q,lengths,time=0.):
    """
    Get the resulting torques acting on the catheter at this instant in time
    
    Inputs:
        n - number of elments
        q - generalized coordinates
        lengths - lengths of each elements
        time - current time in the simulation
    
    Outputs:
        tau - torques along each hinge in the Z direction
    """
    
    # Get the x and y coordinates of the beam
    x,y = get_xy_coords(n,q,lengths)

    # For each origin point of each link, integrate the cross product of the 
    # position vector for the link with the  force converted into the link frame.
    # Return the torques in a way that can be added to the state variables when
    # integrating

    tau = np.zeros(n)
    
    # for each element
    for i in range(n):
        
        # Get the displacement of this element
        angles = np.atleast_2d(q[i])
        
        # Because we want to rotate from the global coordinate frame into 
        # the local frame for each element, we will use the negative rotation
        # angle in our calculations
        angles = -angles

        frame_origin = np.array([[x[0,i]],
                                 [y[0,i]]])

        # Direction vector pointing along the X-direction of the desired link
        # This vector is defined in the global coordinate frame
        r = np.array([[x[0,i] - x[0,i-1]],
                      [y[0,i] - y[0,i-1]]])

        # Unit vector pointing in the direction aligned with the link
        # This vector is divided by L, which is the magnitude of r
        u = r / lengths[i]
        
        # Get the projected force on the element at position "x"
        proj_force = lambda x: body_rotation(force(x * u + frame_origin,r,time),angles)
        
        # Compute the torque applied by the force at position "x"
        torquez = lambda x: np.cross(np.array([x,0]),np.array([proj_force(x)]).flatten())
        
        # The total torque applied to the element is the integral of the torque
        # across its length
        torques = np.array([integrate.quad(torquez,0,lengths[i])[0]])

        tau[i] = torques
        
    return tau

def nonlinear_response(n,args,funcs,control_args):
    """
    Create a response from the nonlinear equations of motion
    
    Inputs:
        n - number of elements
        args - response-specific arguments
            f_max - maximum force allowed by the actuator
            time - time array for the simulation
            X0 - initial values for the catheter model
            y_desired - desired tip location for the y-coordinate 
        funcs - nonlinear parameterized functions
            mm_func - mass matrix function
            fo_func - forcing matrix function
        control_args - arguments regarding control
            Kp - proportional gain
            Kd - derivative gain
            use_control - option to use PD control or let the catheter move uncontrolled
    
    Outputs:
        response - integrated equations of motion
    """
    
    # Unpack arguments
    f_max, time, X0, y_desired,lengths = args
    kp,kd,use_control = control_args
    mm_func,fo_func = funcs
   
    # Define the initial conditions
    q_i = np.array(X0[:n])
    u_i = np.array(X0[n:])

    # initial positions and velocities – assumed to be given in degrees
    y0 = np.concatenate([np.tile(q_i,n),np.tile(u_i,n)])
    
    tau = np.zeros(n)
    force_in = np.zeros(1)
    
    # function which computes the derivatives of parameters
    def gradient(x, currtime,t_sys,tau,force_in):
        
        # Use this option to show a controlled or uncontrolled catheter
        if use_control:
            
            # Get the current x and y coordinates of the catheter
            curr_x,curr_y = get_xy_coords(n,x,lengths)
            curr_x_dot,curr_y_dot = get_xy_deriv(n,x,lengths)

            # Use proportional-Derivative control on the tip y-coordinate
            force_in[0] = kp * (y_desired - curr_y[0,-1]) + kd * (0 - curr_y_dot[0,-1])
        else:
            force_in[0] = 0.
        
        # Get forces from the external disturbance
        tau = get_forces(n,x,lengths,currtime)
        
        # Solve the equations at this time step
        vals = np.concatenate((x, tau, force_in))
        sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))

        return np.array(sol).T[0]

    return odeint(gradient, y0, time, args=(time,tau, force_in))

def linear_response(n,A,B,args,control_args):
    """
    Create a response from the linear equations of motion
    
    Inputs:
        n - number of elements
        A - A matrix for the linear equations of motion
        B - B matrix for the linear equations of motion
        args - response-specific arguments
            f_max - maximum force allowed by the actuator
            time - time array for the simulation
            X0 - initial values for the catheter model
            y_desired - desired tip location for the y-coordinate 
        control_args - arguments regarding control
            Kp - proportional gain
            Kd - derivative gain
            use_control - option to use PD control or let the catheter move uncontrolled
    
    Outputs:
        response - integrated equations of motion
    """
    
    # Unpack arguments
    f_max, time, X0, y_desired,lengths = args
    kp,kd,use_control = control_args

    # Create the initial conditions
    q_i = np.array(X0[:n])
    u_i = np.array(X0[n:])

    # initial positions and velocities – assumed to be given in degrees
    y0 = np.concatenate([np.tile(q_i,n),np.tile(u_i,n)])
    
    tau = np.zeros(n)
    force_in = np.zeros(B.shape[1])

    # function which computes the derivatives of parameters
    def gradient(x, currtime, t_sys):

        # Use this option to show a controlled or uncontrolled catheter
        if use_control:
            
            # Get the current x and y coordinates of the catheter
            curr_x,curr_y = get_xy_coords(n,x,lengths)
            curr_x_dot,curr_y_dot = get_xy_deriv(n,x,lengths)

            # Use proportional-Derivative control on the tip y-coordinate
            force_in[0] = kp * (y_desired - curr_y[0,-1]) + kd * (0 - curr_y_dot[0,-1])
        else:
            force_in[0] = 0.
        
        # Get torques from the external disturbance
        tau = get_forces(n,x,lengths,currtime)
        
        # apply the external torques to the input vector
        force_in[1:] = tau[1:]
       
        # Solve the equation of motion
        sol = np.matmul(A,x) + np.matmul(B,force_in)
        
        return np.array(sol)

    return odeint(gradient, y0, time, args=((time,)))