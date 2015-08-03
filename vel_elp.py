# For analytical derivation refer to vtilt.nb
# 
# PURPOSE:
#       For star catalog (l,b,d,vlos) where d and vlos can be pdfs,
#       for each proposed value of velocity dispersion matrix (3X3 non-diagonal) and vrot
#       returns marginalized/unmarginalised Velocity Ellipsoid Distribution 
#       and associated likelihood.
####################################################################

import numpy as np
import scipy.integrate as sint
import scipy.stats as st

#+++++ Few utilities++++++++++++++++++++++++++++++++++++++
def ttrans_metric(th,fi):
    #Cartesian -> Galactocentric spherical 

    #Cartesian vector to Spherical Polar coordinates conversion metric
    Cos   = np.cos
    Sin   = np.sin 
    Sinth = Sin(th)
    Costh = Cos(th) 
    Sinfi = Sin(fi)
    Cosfi = Cos(fi)

    Tcx2sp = [[Sinth*Cosfi ,Sinth*Sinfi , Costh],
             [ Costh*Cosfi  ,Costh*Sinfi ,-Sinth],
             [ -Sinfi       ,Cosfi       , 0.0]]
    return np.array(Tcx2sp)

def trans_metric(l,b):
    #Galactic spherical -> cartesian

    #lati-longi cordinates to cartesian coordinates conversion metric
    #To compensate for the negative 'vb' vector I have put negative in
    #column IInd of the metric itself. 
    Cos   = np.cos
    Sin   = np.sin 
    Cosb = Cos(b)
    Cosl = Cos(l)
    Sinb = Sin(b)
    Sinl = Sin(l)

    Tlb2cx = [[Cosb*Cosl ,-Sinb*Cosl ,-Sinl],
              [Cosb*Sinl ,-Sinb*Sinl ,Cosl],
              [Sinb      ,+Cosb      ,0.0]]
    return np.array(Tlb2cx)

def eta_matrix(l,b,d,Rcen):
    """
    l,b in radians
    d is distance
    Rcen = scalar or a list [xcen, ycen, zcen]
    """
    Tlb2cx  = trans_metric(l, b)     
    x       = d*Tlb2cx[0,0]
    y       = d*Tlb2cx[1,0]
    z       = d*Tlb2cx[2,0]

    if np.isscalar(Rcen): 
        xgc, ygc, zgc = x-Rcen, y, z
    else:
        xgc, ygc, zgc = x-Rcen[0], y-Rcen[1], z-Rcen[2]
 
    rgc     = np.sqrt(xgc**2 + ygc**2 + zgc**2)
    theta   = np.arccos(zgc/rgc)
    phi     = np.arctan2(ygc,xgc)
    Transm  = ttrans_metric(theta, phi)
    FM_out  = np.dot(Transm, Tlb2cx)
    return np.array(FM_out.tolist()), rgc
#+++++ Few utilities++++++++++++++++++++++++++++++++++++++

#+++++ Few utilities++++++++++++++++++++++++++++++++++++++
def err_pdf( loc, scale, size=15, cutoff=5):
    """
    PURPOSE
    --------
    Given, loc and scale this returns gaussian pdf that can be used as a err_pdf.
    
    INPUT
    ------
    loc    = [data] dim.
    scale  = standard deviation. [data] dim 
    size   = 15 #refinement of the grid. For small cutoff use larze size and viceversa.
    cutoff = 3 #Grid is created between +-5*sigma (here, scale) on either side of loc

    OUTPUT
    -------
    Gaussian(x) at corresponding x
    """
    if cutoff<3:
        print 'Are you sure you just want to use <+-3sigma value on either side?'
    grid    = loc + scale*np.linspace(-cutoff, cutoff, size)[:,None].repeat(loc.shape[0],axis=1)
    err_pdf = st.norm.pdf( grid, loc=loc, scale=scale)
    return err_pdf, grid
#+++++ Few utilities++++++++++++++++++++++++++++++++++++++

#+++++ Velocity Ellipsoid Class+++++++++++++++++++++++++++
class VEL_ELP:
    """
    PURPOSE
    ---------
        Returns velocity ellipsoid model for each star 
                   a. Unmarginalized case
                   b. Marginalized case. Marginalisation over (vl,vb)
                        -> Analytical expression in marginalization 
                           is subjected to a condition [4T1>T3^2/T2].
                           If the condition is not met numerical integration of 
                           unmarginalized case must be done (FEATURE ABSENT AT A MOMENT). 
              
    """
    def __init__(self, eta, vlos, tauj, vrotj):
        """
        PURPOSE
        --------

        DATA:
            eta  = coordinate matrix of each star [3X3Xdata] or [3X3Xd_pdfXdata] dimension
                  -> Use function eta_matrix to create eta
            vlos = [data] or [vlos_pdfXdata]

        MODEL PARAMETERS:
            tauj = [3X3], inverse of the dispersion tensor 
                     (\Sigma^-1 term in a multivariate gaussian distribution)
            vroti= a scalar
        """

        self.eta  = eta
        if eta.ndim>3 and vlos.ndim>1:
            print 'Adding extra dimension in vlos for numpy compatibility' 
            self.vlos = vlos[:,None]
        else:
            self.vlos = vlos
        self.tauj  = tauj
        self.vrotj = vrotj
       
        self.eta00 = eta[0,0]
        self.eta01 = eta[0,1]
        self.eta02 = eta[0,2]

        self.eta10 = eta[1,0]
        self.eta11 = eta[1,1]
        self.eta12 = eta[1,2]

        self.eta20 = eta[2,0]
        self.eta21 = eta[2,1]
        self.eta22 = eta[2,2]

        self.tau00 = tauj[0,0]
        self.tau01 = tauj[0,1]
        self.tau02 = tauj[0,2]

        self.tau10 = tauj[1,0]
        self.tau11 = tauj[1,1]
        self.tau12 = tauj[1,2]

        self.tau20 = tauj[2,0]
        self.tau21 = tauj[2,1]
        self.tau22 = tauj[2,2]

    def T1( self ):
        return self.eta02**2*self.tau00 + self.eta12**2*self.tau11 + self.eta02*( self.eta12*(self.tau01 + self.tau10) + self.eta22*( self.tau02 + self.tau20)) + self.eta12*self.eta22*(self.tau12 + self.tau21)+ self.eta22**2.*self.tau22


    def T2( self):
        return self.eta01**2*self.tau00 + self.eta11**2*self.tau11 + self.eta01*( self.eta11*( self.tau01 + self.tau10 ) + self.eta21*( self.tau02 + self.tau20)) + self.eta11*self.eta21*( self.tau12 + self.tau21)+ self.eta21**2.*self.tau22

    def T3(self):
        return 2*self.eta11*self.eta12*self.tau11 + self.eta12*self.eta21*self.tau12 + self.eta11*self.eta22*self.tau12 + self.eta02*(self.eta11*(self.tau01+self.tau10)+self.eta21*(self.tau02 + self.tau20))+self.eta01*(2*self.eta02*self.tau00 + self.eta12*(self.tau01+self.tau10) + self.eta22*(self.tau02+self.tau20))+ self.eta12*self.eta21*self.tau21+ self.eta11*self.eta22*self.tau21 + 2*self.eta21*self.eta22*self.tau22


    def T4(self):
        return 2*self.vlos*self.eta10*self.eta12*self.tau11 - self.vrotj*self.eta12*self.tau12 + self.vlos*self.eta12*self.eta20*self.tau12 + self.vlos*self.eta10*self.eta22*self.tau12 + self.eta02*( self.vlos*self.eta10*(self.tau01 + self.tau10 )- ( self.vrotj- self.vlos*self.eta20)*(self.tau02+ self.tau20)) + self.vlos*self.eta00*(2*self.eta02*self.tau00 + self.eta12*(self.tau01 + self.tau10)+ self.eta22*(self.tau02+ self.tau20)) -self.vrotj*self.eta12*self.tau21 + self.vlos*self.eta12*self.eta20*self.tau21 + self.vlos*self.eta10*self.eta22*self.tau21 - 2*self.vrotj*self.eta22*self.tau22 + 2*self.vlos*self.eta20*self.eta22*self.tau22

    def T5(self):
        return 2*self.vlos*self.eta10*self.eta11*self.tau11-self.vrotj*self.eta11*self.tau12+self.vlos*self.eta11*self.eta20*self.tau12+self.vlos*self.eta10*self.eta21*self.tau12 +self.eta01*(self.vlos*self.eta10*(self.tau01+self.tau10)-(self.vrotj-self.vlos*self.eta20)*(self.tau02+self.tau20))+self.vlos*self.eta00*(2*self.eta01*self.tau00+self.eta11*(self.tau01 +self.tau10)+self.eta21*(self.tau02+self.tau20))-self.vrotj*self.eta11*self.tau21+self.vlos*self.eta11*self.eta20*self.tau21+self.vlos*self.eta10*self.eta21*self.tau21-2*self.vrotj*self.eta21*self.tau22+2*self.vlos*self.eta20*self.eta21*self.tau22
 
    def T6(self):
        return self.vlos**2.*self.eta00**2.*self.tau00 + self.vlos**2.*self.eta10**2.*self.tau11 + self.vlos*self.eta00*( self.vlos*self.eta10*(self.tau10 + self.tau10 )- ( self.vrotj-self.vlos*self.eta20 )*(self.tau02 + self.tau20))+ self.vlos*self.eta10*(-self.vrotj+self.vlos*self.eta20)*(self.tau12 + self.tau21)+(self.vrotj-self.vlos*self.eta20)**2*self.tau22

    def marg_df( self, tau_det ):
        """
        INPUT
        ---------
            tau_det = Determinant of tau
        PURPOSE
        ---------
            Returns likelihood of each data point. 
                 Marginalized DF. case with marginalisation over (vl, vb)   
        OUTPUT
        --------
        """

        selfT1 = self.T1()
        selfT2 = self.T2()
        selfT3 = self.T3()
        selfT4 = self.T4()
        selfT5 = self.T5()
        selfT6 = self.T6()

        if np.any(4*selfT1 - (selfT3**2/selfT2))<0.:
            raise ValueError('Condition [4 T1>= T3^2/T2] unmet. Numerically integrate unmarg_kern')
        expnumer = selfT2*selfT4**2 - selfT3*selfT4*selfT5 + selfT1*selfT5**2
        expdenom = 4*selfT1*selfT2 - selfT3**2
        denom    = np.sqrt((4*selfT1*selfT2 - selfT3**2)*((2*np.pi)**3.*tau_det ))

        expo_arg = 0.5*(expnumer/expdenom - selfT6)
        if np.any(np.abs(expo_arg))>200:
            raise ValueError('The argument of expo term blows up to inf/0.')
        return (4.*np.pi/denom)*np.exp( expo_arg )

    def unmarg_df( self, vl, vb, tau_det ):
        """
        INPUT
        ------
            vl, vb  = galactic coordinates (heliocentric, solar motion subtracted)
            tau_det = Determinant of tau
        PURPOSE
        --------
         Returns likelihood of each data point. 
                 *Unmarginalized DF. case  (hence vl, vb need to be provided)
                 *Identical to multivariate gaussian with loc = {0,0,vrot} and scale = Inverse[tauj] 
        """
        selfT1 = self.T1()
        selfT2 = self.T2()
        selfT3 = self.T3()
        selfT4 = self.T4()
        selfT5 = self.T5()
        selfT6 = self.T6()
        kern = selfT1*vl**2 + selfT2*vb**2 + selfT3*vl*vb + selfT4*vl + selfT5*vb + selfT6
        denom= np.sqrt( (2*np.pi)**3.*tau_det )

        expo_arg = -0.5*kern
        if np.any(np.abs(expo_arg))>200:
            raise ValueError('The argument of expo term blows up to inf/0.')
        return np.exp( expo_arg )/denom

    def err_conv( self, df, err_pdf, grid ):
        """
        PURPOSE
        --------
            Returns DF convolved with distance/velocity or both pdfs
                 DF can be marginalized or unmarginalised
        INPUT
        ------
            df       = [dist or vlos_grid X data] | [ vlos X dist X data ]
            err_pdf  = dict {'vlos':, 'd':}
                       [dist or vlos_grid X data] | [ dist_grid X data, vlos_grid X data] 
            grid     = dict {'vlos':, 'd':} 
                       [dist or vlos_grid X data] | [ dist_grid X data, vlos_grid X data]
        """
        if df.ndim==2:
            #return np.trapz( df*err_pdf, x=grid, axis=0 )
            return sint.simps( df*err_pdf, x=grid, axis=0 )
 
        elif df.ndim==3:
            if (df.shape[0]!= err_pdf['vlos'].shape[0])|(df.shape[0]!= grid['vlos'].shape[0]):
                raise ValueError('DF must be [vlosXdistXdata] & vlos_pdf must be [vlosXdata] dim.')
            if (df.shape[1]!= err_pdf['d'].shape[0])|(df.shape[1]!= grid['d'].shape[0]):
                raise ValueError('DF must be [vlosXdistXdata] & dist_pdf must be [distXdata] dim.')
            #return np.trapz( np.trapz( df*err_pdf['vlos'][:,None], x=grid['vlos'][:,None], axis=0 )*err_pdf['d'], x=grid['d'], axis=0)
            return sint.simps( sint.simps( df*err_pdf['vlos'][:,None], x=grid['vlos'][:,None], axis=0 )*err_pdf['d'], x=grid['d'], axis=0)

    def likelihood( self, df):
        """
        PURPOSE
        --------
            Returns the likelihood given a (marginalised/unmarginalised) df function
        """
        if df.ndim!=1:
            raise ValueError('Are you sure DF is not conditional.DF should be [data] dim.')
        return np.sum( np.log(df) )
#+++++ Velocity Ellipsoid Class+++++++++++++++++++++++++++

if __name__=='__main__':
    import ebf
    import numpy.linalg as lg
    import scipy.stats as st
    import numpy.random as rd
    rd.seed(5)

    fn = ebf.read('/home/prajwal/worka/PhD/modules/gve_mock/mock_tan_vrot.ebf')
    #print fn['info']
    k = 10000 #Use only k star particles.
    
    #ERROR CONVOLUTION OPTION
    verr_conv = 'n'
    derr_conv = 'n'
    verr_derr_conv = 'n'
             
    #Data+++++++++++++++
    k = rd.permutation(np.arange(0,fn['dhc'].shape[0],1))[:k]

    vl, vb, vlos = fn['vlhc'][k], fn['vbhc'][k], fn['vhhc'][k]
    vr, vtheta, vphi = fn['vr'][k], fn['vtheta'][k], fn['vphi'][k]
    
    l, b, d = fn['lhc'][k], fn['bhc'][k], fn['dhc'][k]
    r, theta, phi= fn['r'][k] , fn['theta'][k] , fn['phi'][k]

    #Instead of scalar d use pdf for d
    if derr_conv=='y' or verr_derr_conv=='y':
        derr     = 0.1 
        dist_pdf, d = err_pdf( d, np.abs(d)*derr, size=15, cutoff=5 )
    #Instead of scalar vlos use pdf for vlos
    if verr_conv=='y' or verr_derr_conv=='y':
        verr     = 0.1 
        vlos_pdf, vlos = err_pdf( vlos, np.abs(vlos)*verr, size=15, cutoff=5)
    #Data+++++++++++++++

    eta, rgc = eta_matrix( l, b, d, 8.5)

    COV    = np.array([[ np.var(vr), 0, 0],[0, np.var(vtheta), 0.],[0, 0, np.var(vphi)]])
    INVCOV = lg.inv(COV)
    DETCOV = lg.det(COV)

    LOC    = np.array([np.mean( vr ), np.mean( vtheta), np.mean( vphi )])
    
    VEL      = np.array([vr, vtheta, vphi]).T

    import time

    velp     = VEL_ELP( eta, vlos, INVCOV, LOC[2] )

    t0 = time.time()

    #++++++ Unmarginalised case+++++++++++++++++++++++++++
    #Ideally in an unmarginalised + errors unconvolved case gaussian shud be same as velpkern
    gaussian = st.multivariate_normal.pdf( VEL, LOC, COV)

    velpkern = velp.unmarg_df( vl, vb, DETCOV )

    if verr_derr_conv=='y':
        velpkern_conv = velp.err_conv( velpkern,{'vlos':vlos_pdf,'d':dist_pdf},{'vlos':vlos,'d':d})
    elif verr_conv=='y':
        velpkern_conv = velp.err_conv( velpkern, vlos_pdf, vlos )
    elif derr_conv=='y':
        velpkern_conv = velp.err_conv( velpkern, dist_pdf, d )
    t1 = time.time()
    print '%1.2f'%(t1-t0)
    #++++++ Unmarginalised case+++++++++++++++++++++++++++

    #++++++ Marginalised case+++++++++++++++++++++++++++
    velpkern_marg = velp.marg_df( DETCOV )

    if verr_derr_conv=='y':
        velpkern_marg_conv = velp.err_conv( velpkern_marg, {'vlos':vlos_pdf,'d':dist_pdf}, {'vlos':vlos,'d':d})
    elif verr_conv=='y':
        velpkern_marg_conv = velp.err_conv( velpkern_marg, vlos_pdf, vlos )
    elif derr_conv=='y':
        velpkern_marg_conv = velp.err_conv( velpkern_marg, dist_pdf, d )
    t2 = time.time()
    print '%1.2f'%(t2-t1)
    #++++++ Marginalised case+++++++++++++++++++++++++++
    if velpkern.ndim>1:
        print '%1.3e'%velp.likelihood( velpkern_conv ), '%1.3e'%velp.likelihood( velpkern_marg_conv )
    else:
        print '%1.3e'%velp.likelihood( velpkern ), '%1.3e'%velp.likelihood( velpkern_marg )


    import matplotlib.pyplot as plt
    plt.figure()
    if velpkern.ndim>1:
        plt.plot( gaussian, velpkern_conv, 'g.', alpha=0.1)
    else:
        plt.plot( gaussian, velpkern, 'g.', alpha=0.1)
    plt.plot( gaussian, gaussian, 'r--')
    plt.show()
