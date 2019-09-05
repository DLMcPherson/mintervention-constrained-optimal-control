import numpy as np
import numpy.linalg as LA

# Represents function of the quadratic form:
#    $ x^T Z x + x^T z + c $
class QuadraticForm:
    """Bundles matrix for bilinear terms, vector for linear terms, and scalar for constant offset of quadratic form

    Attributes:
        Z (np.matrix): Matrix representing purely quadratic terms
        z (np.matrix): Column vector representing linear terms
        c (np.matrix): Scalar representing constant offset
    """
    def setQuadraticTerm(self,_matrix):
        self.Z = np.matrix(_matrix)

    def setLinearTerm(self,_vector):
        self.z = np.matrix(_vector)

    def setConstantTerm(self,_scalar):
        self.c = np.matrix(_scalar)

    def evaluate(self,at):
        return(np.dot(np.dot(np.transpose(at),self.Z),at) + np.dot(np.transpose(at),self.z) + self.c)

    def derivative(self,at):
        return(self.z + 2*self.Z*np.matrix(at).T)

    def hessian(self):
        return(2*self.Z)

    def __add__(self, other):
        if(self.dimension == other.dimension):
            return QuadraticForm(self.dimension,self.Z+other.Z,self.z+other.z,self.c+other.c)
        else:
            raise ValueError('Dimension of two quadratics to be added do not match')
            return

    def __mul__(self, scalar):
        return QuadraticForm(self.dimension, self.Z*scalar, self.z*scalar, self.c*scalar)

    def __init__(self,dimension,_Z,_z,_c):
        """Initializes quadratic form

        Args:
            Z (array_like): Matrix representing purely quadratic terms
            z (array_like): Column vector representing linear terms
            c (array_like): Scalar representing constant offset
        """
        self.dimension = dimension
        self.Z = np.matrix(_Z)
        self.z = np.matrix(_z)
        self.c = np.matrix(_c)

def onestep_dlqr_affine(A,B,Q,R,X,lamb = 0):
    """Perform one Bellman backup on the cost-to-go quadratic for a discrete-time LQR problem

    Assuming a dynamic system:
      x[k+1] = A x[k] + B u[k]

    with objective functional:
      J[u()] = \sum x[k]^T*Q*x[k] + q^T*x[k] + c_q + u[k]^T*R*u[k] + u[k]^T*r + c_r

    Args:
        A (np.matrix): Dynamic flow matrix
        B (np.matrix): Control dynamic coefficient matrix
        Q (QuadraticForm): State-dependent running cost
        R (QuadraticForm): Control-dependent running cost
        X (QuadraticForm): Cost-to-go matrix at current time step

    Returns:
        K (np.matrix): Optimal control feedback gain matrix
        k (np.matrix): Optimal control feedforward vector
        Xprime (QuadraticForm): Updated cost-to-go matrix
        eigVals (np.array): Open-loop transfer function eigenvalues
    """
    # Ensure all inputs are in the correct format
    A = np.matrix(A) ; B = np.matrix(B)

    # Invert the pseudo-Hamiltonian matrix
    V_uu_evals, V_uu_evecs = LA.eig(R.Z+B.T*X.Z*B)
    V_uu_evals[V_uu_evals < 0.0] = 0.0 # Ensure positive semi-definiteness
    #V_uu_evals[V_uu_evals > 1000000000000.0] = 1000000000000.0 # Clip eigenvalues from growing unbounded
    V_uu_evals += lamb # Levenberg-Marquadt Regularization
    V_uu_inv = np.dot(V_uu_evecs,np.dot(np.diag(1.0/V_uu_evals), V_uu_evecs.T))
    #V_uu_inv = LA.inv(R.Z+B.T*X.Z*B) # Alternative: Direct inversion

    # Calculate the control laws (feedback and feedforward terms)
    K = -np.matrix(V_uu_inv*(B.T*X.Z*A))
    k = -0.5*np.matrix(V_uu_inv*(B.T*X.z + R.z))

    # Update the cost-to-go based on following this new control for one time step
    Abbok = (A+B*K)
    Xprime = QuadraticForm(X.dimension,0,0,0)
    Xprime.setQuadraticTerm( Q.Z + K.T*R.Z*K + Abbok.T*X.Z*Abbok )
    Xprime.setLinearTerm( K.T*R.z + Q.z + Abbok.T*X.z + 2*Abbok.T*X.Z*B*k + 2*K.T*R.Z*k)
    Xprime.setConstantTerm( R.c+Q.c+X.c+k.T*B.T*X.Z*B*k+k.T*R.Z*k+k.T*R.z+k.T*B.T*X.z )

    eigVals, eigVecs = LA.eig(A+B*K)

    return K, k, Xprime, eigVals
