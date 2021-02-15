import json
import numpy as np
import itertools

class LevelSetFunctionABC():
    def __init__(self):
        self.dim = 1

    def value(self,queryState):
        """Returns the value of the level set at the given query state

        Args:
            queryState (np.array): state at which to evaluate the cost function

        Returns:
            interpolatedValue (float): value multilinearly interpolated from neighboring gridcells
        """
        return(0)

    def gradient(self,queryState):
        """Returns the gradient of the value function at the given query state

        Args:
            queryState (np.array): state at which to evaluate the gradient

        Returns:
            gradient (np.array): gradient of the value function at the query state
        """
        gradient = np.zeros(self.dim)
        return(gradient)

class GriddedLevelSetFunction(LevelSetFunctionABC):

    """
    Encapsulates a multidimensional function calculated using the LSToolbox in
    MATLAB by Dr. Ian Mitchell. These are used in this code as reachable/safety
    sets for verifying constraint satisfaction.
    """

    def __init__(self,filename):
        # load the level set JSON exported from MATLAB via json_export_reachset()
        with open(filename) as f:
            loadedLS = json.load(f)
        self.gmax = np.array(loadedLS["gmax"])
        self.gmin = np.array(loadedLS["gmin"])
        self.gdx  = np.array(loadedLS["gdx"] )
        self.gN   = np.array(loadedLS["gN"]  )
        self.gperiodicity = np.array(loadedLS["gperiodicity"])
        self.dim = self.gmax.size
        self.data = np.array(loadedLS["data"])

    def inGridIndex(self,queryState):
        """Converts a global state into coordinates in the grid's frame and units

        Args:
            queryState (np.array): the state for converting into grid units

        Returns:
            indexedState (np.array): the converted coordinates within the grid
        """
        indexedState = (queryState - self.gmin ) / self.gdx
        #indexedState = indexedState - (indexedState >= self.gN) * self.gN + (indexedState < 0) * self.gN
        return(indexedState)

    def value(self,queryState):
        """Returns the value of the level set at the given query state

        Args:
            queryState (np.array): state at which to evaluate the cost function

        Returns:
            interpolatedValue (float): value multilinearly interpolated from neighboring gridcells
        """
        queryIndex = self.inGridIndex(queryState)
        interpolatedValue = 0.0
        for corner in itertools.product(*zip([0]*self.dim,[1]*self.dim)):
            cornerIndex = np.floor(queryIndex) + np.array(corner)
            weight = np.prod(1-np.abs(queryIndex - cornerIndex))
            interpolatedValue += weight*self.indexedValue(cornerIndex)
        return(interpolatedValue)

    def gradient(self,queryState):
        gradient = np.zeros(self.dim)
        for ii,difference in enumerate(self.gdx):
            offset = np.zeros(self.dim)
            offset[ii] = difference/2
            gradient[ii] = (self.value(queryState + offset) - self.value(queryState - offset))/difference
        return(gradient)

    def hessian(self,queryState):
        hessian = np.zeros((self.dim,self.dim))
        for ii,difference in enumerate(self.gdx):
            offset = np.zeros(self.dim)
            offset[ii] = difference/2
            hessian[ii] = (self.gradient(queryState + offset) - self.gradient(queryState - offset))/difference
        hessian = 0.5 * (hessian + hessian.transpose())
        return(hessian)

    def indexedValue(self,queryIndex):
        """Returns the value of the level set at the given query state

        Args:
            queryIndex (np.array): index at which to evaluate the cost function

        Returns:
            value (float): value at the indexed gridpoint
        """
        index = np.copy(queryIndex)
        # Mind the grid's boundaries
        for ii,periodicity in enumerate(self.gperiodicity):
            maxIndex = self.gN[ii]
            if periodicity: # Wrap periodically if there's a periodic state (e.g. angle)
                #index[ii] = index[ii] - (index[ii] >=  maxIndex) *  maxIndex + (index[ii] < 0) *  maxIndex
                index[ii] = index[ii] % maxIndex
            else: # Project to the border if outside the bounds
                if (index[ii] >=  maxIndex):
                    index[ii] = maxIndex - 1
                if (index[ii] < 0):
                    index[ii] = 0
        # Read out the value
        #print(tuple(index.astype(int)))
        value = self.data[tuple(index.astype(int))]
        #value = self.data[index.astype(int)]
        return(value)

class AnalyticDoubleIntegratorLS(LevelSetFunctionABC):
    def __init__(self,_center,_width,_leeway):
        self.dim = 2

        self.obstacleCenter = _center
        self.obstacleWidth = _width
        self.controlLeeway = _leeway

    def value(self,queryState):
        """Returns the value of the level set at the given query state

        Args:
            queryState (np.array): state at which to evaluate the cost function

        Returns:
            interpolatedValue (float): value multilinearly interpolated from neighboring gridcells
        """
        position = queryState[0]
        velocity = queryState[1]
        # If your velocity and displacement from the obstacle are in the same
        # direction, you're at risk of a crash:
        displacement = self.obstacleCenter - position
        if velocity * displacement > 0:
          return(  np.abs(displacement)-self.obstacleWidth
                 - np.power(velocity,2)/(2.0*self.controlLeeway) )
        else:
            return(np.abs(displacement)-self.obstacleWidth)

    def gradient(self,queryState):
        """Returns the gradient of the value function at the given query state

        Args:
            queryState (np.array): state at which to evaluate the gradient

        Returns:
            gradient (np.array): gradient of the value function at the query state
        """
        gradient = np.zeros(self.dim)

        position = queryState[0]
        velocity = queryState[1]
        displacement = self.obstacleCenter - position
        # Compute the gradient with respect to position through the abs
        if displacement > 0:
          gradient[0] = -1
        else:
          gradient[0] =  1
        # Compute the gradient with respect to velocity through the squared min
        if velocity * displacement > 0:
          gradient[1] = - velocity/self.controlLeeway
        else:
          gradient[1] = 0

        return(gradient)

class DecoupledLevelSetFunction(LevelSetFunctionABC):

    """
    Encapsulates a decoupled level set in the fashion of Herbert and Chen.
    The state space is partitioned into two decoupled sub-spaces each with their
    own safe sets.
    For this code, the decoupled sub-spaces' states must have continguous indexes.
    """

    def __init__(self,_setA,_setB):
        self.setA = _setA
        self.dimA = _setA.dim
        self.setB = _setB
        self.dimB = _setB.dim

        self.dim = _setA.dim + _setB.dim

        #self.gmax = np.concatenate((_setA.gmax,_setB.gmax))
        #self.gmin = np.concatenate((_setA.gmin,_setB.gmin))
        #self.gdx  = np.concatenate((_setA.gdx,_setB.gdx))
        #self.gN   = np.concatenate((_setA.gN,_setB.gN))
        #self.gperiodicity = np.concatenate((_setA.gperiodicity,_setB.gperiodicity))

    def value(self,queryState):
        """Returns the value of the level set at the given query state

        Args:
            queryState (np.array): state at which to evaluate the cost function

        Returns:
            interpolatedValue (float): value multilinearly interpolated from neighboring gridcells
        """
        valueA = self.setA.value(queryState[0:self.dimA])
        valueB = self.setB.value(queryState[self.dimA:self.dim])
        if valueA > valueB:
            return(valueA)
        else:
            return(valueB)

    def gradient(self,queryState):
        gradient = np.zeros(self.dim)

        valueA = self.setA.value(queryState[0:self.dimA])
        valueB = self.setB.value(queryState[self.dimA:self.dim])
        if valueA > valueB:
            gradient[0:self.dimA] = self.setA.gradient(queryState[0:self.dimA])
        else:
            gradient[self.dimA:self.dim] = self.setB.gradient(queryState[self.dimA:self.dim])
        return(gradient)
