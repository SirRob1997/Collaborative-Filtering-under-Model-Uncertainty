import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import polytope as pc
import cvxpy as cvx
import cvxopt
from cvxopt import matrix
import scipy
import scipy.optimize

import time
import logging

################################## User-centric Reachability ##################################

def get_optimal_actions_cvx(H, target_item, 
                        mutable_items, immutable_items, ratings,
                        bias = None,
                        l2_reg=0., rating_bounds=(0,5)):
    import cvxpy as cvx
    '''
    Constructs and solves the one-step reachability problem for regularized 
    matrix factorization:
        min  |u|_1
        s.t. select(model(r+u)) = target_item
    Assumes ratings must be constrained to [-rating_bound, rating_bound].
    '''
    n_latent_features, n_items = H.shape

    # translating to latent space
    ## p = v_0 + A u
    full_support = np.concatenate([immutable_items,mutable_items]).astype(int)
    Q = H.T
    B = Q[full_support].T.dot(Q[full_support])+l2_reg*np.eye(n_latent_features)
    Binv = scipy.linalg.inv(B) # TODO: faster?
    A = Binv.dot(Q[mutable_items].T) 

    r_vec = np.zeros(n_latent_features)
    if len(immutable_items) > 0: # assume this means not a history edit - reaction
        r_vec += Q[immutable_items].T.dot(ratings)
    if bias is not None:
        item_bias, user_bias, b0 = bias
        r_vec -= Q[full_support].T.dot(item_bias[full_support]+user_bias+b0)
    v0 = Binv.dot(r_vec)
    
    # constructing target region
    ## matrices for region such that select(v_{t+1}) = target_item
    A_reg, b_reg = item_factors_to_inequality_constraints(H, target_item, exclude=full_support, tol=1e-2)

    ## A_reg (v + Au) <= b turns into A_reg A u <= b-A_reg v
    con_A = A_reg.dot(A)
    con_b = b_reg - A_reg.dot(v0)

    # variable
    u = cvx.Variable(len(mutable_items))
    cons = [con_A * u <= con_b]
    if rating_bounds is not None:
        lb, ub = rating_bounds
        cons += [u >= lb, u <= ub]

    # COST!
    ## currently: either full history edit or no history edit
    if len(immutable_items) == 0: # assume this means a history edit
        cost = cvx.norm(u-ratings,1)
    else: # reaction -- assume nothing in seen is in mutable
        B_hist = Q[immutable_items].T.dot(Q[immutable_items])+l2_reg*np.eye(n_latent_features)
        if bias is not None:
            item_bias, user_bias, b0 = bias
            r_pred = Q.dot(scipy.linalg.inv(B_hist)).dot(Q[immutable_items].T.dot(ratings+item_bias[immutable_items]+user_bias+b0)) 
        else:
            r_pred = Q.dot(scipy.linalg.inv(B_hist)).dot(Q[immutable_items].T.dot(ratings)) 
        cost = cvx.norm(u-r_pred[mutable_items],1)

    prob = cvx.Problem(cvx.Minimize(cost), cons)
    try:
        prob.solve()
        success = prob.status not in ["infeasible", "unbounded"]
        val = cost.value
    except cvx.error.SolverError:
        success = False
        val = np.inf

    return success, val

################################## Reachability Approximation ########################

def get_latent_aligned_reachable_top_n_items(H, ns, item_bias=None, exclude=None):
    '''
    quick evaluation of reachability by checking whether
    i = argmax_j q_j^T v with v = argmax q_i^T v / |v|
    this is a sufficient but not necessary condition for reachability.

    H: matrix factorization item model
    n: top-n
    item-bias: additional bias in model
    '''

    n_latent_features, n_items = H.shape


    M = H.T.dot(H)
    print('done multiplying')
    if item_bias is not None: M +=  item_bias.reshape(1,-1) # add to each row
    if exclude is not None: M[:,exclude] = -np.inf
    print('starting sort')
    argsort = M.argsort(axis=1)
    print('finished sort')
    ret = []
    for n in ns:
        ret.append(np.unique(argsort[:,-n:]))
    return ret

ZERO_TOL = 1e-6


def get_user_aligned_reachable_top_n_items(H, ns, immutable_items=[], 
                           ratings=[], mutable_items=[], 
                           reg=0, bias=None, constraints=None, changet=False):
    '''
    quick evaluation of reachability by checking whether
        i = argmax_j q_j^T v with v = argmin |q_i - v|
    FORMERLY: v = argmax q_i^T v / |v|
    this is a sufficient but not necessary condition for reachability.

    H: matrix factorization item model
    n: top-n
    immutable_items: indices for which user ratings cannot be changed
    ratings: corresponding ratings
    mutable_item: indices for which users can change ratings
    method: item-base (no user to consider) or not
    reg: l2 regularization in model
    bias: (item-bias, user-bias, b0): additional item/user/overall bias in model
    '''
    n_latent_features, n_items = H.shape

    # TODO: should assert disjoint sets
    full_support = np.concatenate([immutable_items,mutable_items]).astype(int)
    Q = H.T
    B = Q[full_support].T.dot(Q[full_support])+reg*np.eye(n_latent_features)
    Binv = scipy.linalg.inv(B) # TODO: faster?
    A = Binv.dot(Q[mutable_items].T) 

    r_vec = np.zeros(n_latent_features)
    if len(immutable_items) > 0:
        r_vec += Q[immutable_items].T.dot(ratings)
    if bias is not None:
        item_bias, user_bias, b0 = bias
        r_vec -= Q[full_support].T.dot(item_bias[full_support]+user_bias+b0)
    v0 = Binv.dot(r_vec)

    if constraints is None: 
        AAt = A.dot(scipy.linalg.pinv(A))
        M = Q.dot(AAt).dot(Q.T) + Q.dot(v0 - AAt.dot(v0)).reshape(1,-1)
        if bias is not None: 
            item_bias, _, _ = bias
            M += item_bias.reshape(1,-1)

    elif constraints is not None:
        lb, ub = constraints
        if changet:
            test_vs = []
            for i in range(n_items):
                # doesn't seem to make a difference once we add the lower bound of 1
                Als = np.hstack([A, -Q[i][:,np.newaxis]])
                bls = -v0

                bounds=([lb]*len(mutable_items)+[1], [ub]*len(mutable_items)+[np.inf])
                res = scipy.optimize.lsq_linear(Als, bls, bounds=bounds)
                vi = A.dot(res['x'][:len(mutable_items)])-v0
                test_vs.append(vi)
            V = np.array(test_vs).T
        else:
            # TODO: would vectorizing over items be faster in this case?
            # or implement matrix constrained least squares
            test_vs = []
            for i in range(n_items):
                Als = A
                bls = Q[i]-v0

                bounds=(lb,ub)
                res = scipy.optimize.lsq_linear(Als, bls, bounds=bounds)
                vi = A.dot(res['x'])-v0
                test_vs.append(vi)
            V = np.array(test_vs).T
        M = Q.dot(V)
        if bias is not None: 
                item_bias, _, _ = bias
                M += item_bias.reshape(1,-1)

    else:
        M = np.zeros([n_items,n_items]); Q = H.T
        B = Q[full_support].T.dot(Q[full_support])+reg*np.eye(n_latent_features)
        Binv = scipy.linalg.inv(B) # TODO: faster?
        A = Binv.dot(Q[mutable_items].T)

        # QProjA = (Q A) (A.T A)^-1 A.T
        # (A.T A)^-1 A.T = A^dagger = Q^dagger B
        Qdecomp, Rdecomp = np.linalg.qr(A)
        ProjA = Qdecomp.dot(A.T)

        QProjA = Q.dot(ProjA)
        QProjAnorm = np.linalg.norm(QProjA, axis=1)

        r_vec = np.zeros(n_latent_features)
        if len(immutable_items) > 0:
            r_vec += Q[immutable_items].T.dot(ratings)
        if bias is not None:
            item_bias, user_bias, b0 = bias
            r_vec -= Q[full_support].T.dot(item_bias[full_support]+user_bias+b0)
        else:
            item_bias = np.zeros(n_items)
        v0 = Binv.dot(r_vec)

        v0perp = ProjA.dot(v0)-v0
        v0perp_norm = np.linalg.norm(v0perp)
        Qv0perp = Q.dot(v0perp)

        C = Qv0perp + item_bias

        condition1 = C > 0
        condition2 = np.all(v0 == 0) * (item_bias <= 0)
        condition3 = np.logical_not(np.logical_or(condition1, condition2))
        # print(sum(condition1), sum(condition2), sum(condition3))

        if sum(condition1) > 0:
            M[condition1] = C.reshape(1,-1)
        if sum(condition2) > 0:
            M[condition2] = QProjA[condition2].dot(Q.T)

        if sum(condition3) > 0:
            M[condition3] = v0perp_norm * QProjA[condition3].dot(Q.T) + np.abs(C[condition3,np.newaxis]).dot(C[:,np.newaxis].T) / v0perp_norm
            M[condition3] /= np.sqrt(np.power(QProjAnorm[condition3],2)*v0perp_norm**2 +
                                 np.power(C[condition3],2) )[:,np.newaxis]

    M[:,full_support] = -np.inf

    argsort = M.argsort(axis=1)
    ret = []
    for n in ns:
        reachable_items = np.unique(argsort[:,-n:])
        reachable_items = [i for i in reachable_items if i not in full_support]
        ret.append(reachable_items)
    return ret

################################## Item Audit Logic ##################################

def get_top_n_regions(H, n, exclude=[], unit=1, nonnegative=False, verbose=False):
    '''
    H: matrix factorization item model
    n: positive integer
    unit: optional, size of ball for intersection 
    nonnegative: whether MF model is nonnegative

    returns a list of RecTreeNodes
    '''
    n_latent_features, n_items = H.shape
    # ball ensures that regions are valid and bounded
    if nonnegative:
        bound = [0,unit] 
    else:
        bound = [-unit,unit]
    # constructing tree for top-n regions
    ball = pc.box2poly([bound for _ in range(n_latent_features)])
    leaves = [RecTreeNode(ball=(ball.A,ball.b))]
    for i in range(n):
        if verbose: print('Determining depth', i+1, end=' ')
        exclude_arg = [] if i+1<n else exclude
        leaves, ts = add_region_leaves(H, leaves, verbose=verbose, exclude=exclude_arg)
        if verbose: print('found '+str(len(leaves))+' regions, empty checking times: '+str(ts))
    return leaves

def item_factors_to_inequality_constraints(H, target_item, exclude=[], tol=0):
    '''
    Constructs matrices A, b such that
    target_item = argmax_{j notin exclude} (H^T v)_j    <=>    Av <= b 
    optional tol parameter keeps the region away from 0.
    '''
    n_latent, n_items = H.shape
    included_items = [i for i in range(n_items) if (i not in exclude and i != target_item)]
    A = -(H.T[target_item] - H.T[included_items])
    b = np.zeros(A.shape[0])
    # keeping region away from 0
    if tol > 0:
        A, b = modify_polytope_away_from_origin(A, b, tol)
    return A, b

def modify_polytope_away_from_origin(A, b, tol):
    '''
    Constructs polytope matrices A,b that correspond to original pair
    but also remove a section near the origin.
    '''
    avg = np.mean(A, axis=0)
    avg = (avg / np.linalg.norm(avg)).reshape(1,-1)
    A = np.vstack([A, avg])
    b = np.hstack([b,-tol])
    return A, b

def get_item_polytope_matrices(H, exclude=[]):
    '''
    H: matrix factorization item model
    exclude: items to exclude from consideration

    returns polytopes representing regions in which each item is the argmax
    argmax is taken over all items except "exclude"
    '''
    n_latent_features, n_items = H.shape
    item_regions = [None]*n_items
    for i in range(n_items):
        # determining region
        A, b = item_factors_to_inequality_constraints(H, i, exclude=exclude)
        # exclude logic
        item_regions[i] = None if i in exclude else (A, b)
    return item_regions

def add_region_leaves(H, leaves, verbose=False, exclude=[]):
    '''
    H: matrix factorization item model
    leaves: list of RecTreeNode objects

    Returns the next layer of nonempty leaves on the tree
    '''
    new_leaves = []; times = []
    for leaf in leaves:
        region = leaf.get_region_list()
        corresponding_rec = leaf.get_rec_list()
        subregions = get_item_polytope_matrices(H, exclude=corresponding_rec)
        for j,reg in enumerate(subregions):
            if j not in exclude:
                if verbose and j % len(subregions) == 0: print('checking subleaf {}/{}'.format(j+1,len(subregions)))
                new_leaf = RecTreeNode(value=j, region=reg, parent=leaf)
                empty, ts = new_leaf.is_empty()
                times.append(ts)
                if not empty:
                    new_leaves.append(new_leaf)
                    leaf.add_child(new_leaf)
    return new_leaves, np.nanmean(np.array(times), axis=0)

def project_regions_onboarding(H, onboarding_items, leaves, l2_reg=0, mf_type='reg',
                               penalize_only_seen=True, tol=1e-3):
    '''
    Takes a list of RecTreeNode objects and proposed onboarding items
    Removes empty projections onto onboarding subspace
    '''
    n_latent_features, n_items = H.shape
    n_onb = len(onboarding_items)

    if mf_type == 'reg':
        # subspace defined by onboarding picks
        H_onb = H[:,onboarding_items]
        if penalize_only_seen:
            M = np.linalg.inv(H_onb.dot(H_onb.T) + l2_reg * np.eye(n_latent_features)).dot(H_onb)
        else:
            M = np.linalg.inv(H.dot(H.T) + l2_reg * np.eye(n_latent_features)).dot(H_onb)
        def feasible(A, b):
            A_proj = A.dot(M)
            return not check_empty((A_proj, b), method='polytope-fulldim')
    elif mf_type in ['nonneg', 'nonnegative']:
        if penalize_only_seen: print("warning: cannot penalize only seen for NMF")
        if l2_reg > 0: print("warning: cannot do regularized NMF")
        H_onb = H[:,onboarding_items]
        c = np.ones(n_latent_features+n_onb) # np.hstack([np.zeros(n_latent_features), np.ones(n_onb)])
        E = np.zeros([n_items, n_onb])
        for j,i in enumerate(onboarding_items):
            E[i,j] = 1
        A_feas = np.hstack([H.T, -E])
        b_feas = np.zeros(n_items)
        def feasible(A,b):
            # find H.T v = E r, A v <= b 
            G_feas = np.vstack( [np.hstack([A, np.zeros([A.shape[0], n_onb])]),
                                 -np.eye(n_latent_features+n_onb),
                                 np.hstack([np.zeros(n_latent_features), -np.ones(n_onb)])])
            h_feas = np.hstack([b,np.zeros(n_latent_features+n_onb),[-tol]])
            # TODO: need to deal with non-uniqueness!!!!! for n_latent_features > n_onb
            ball = pc.box2poly([[0,1] for _ in range(n_latent_features+n_onb)])
            sol = cvxopt.solvers.lp(c=matrix(c), G=matrix(ball.A), h=matrix(ball.b), # G=matrix(G_feas), h=matrix(h_feas),
                                 A=matrix(A_feas), b=matrix(b_feas), solver='glpk')
            if sol['status'] == 'optimal':
                x = np.array(sol['x'])
                v = x[:n_latent_features].flatten(); r = x[n_latent_features:].flatten()
                print('v=',v, 'r=',r, 'nnls', scipy.optimize.nnls(H_onb.T, r)[0])
            return sol['status'] == 'optimal'
    else:
        # TODO add integer constrains on r
        raise NotImplementedError('method {} not implemented'.format(mf_type))

    # pruning based on projection
    new_leaves = []
    for leaf in leaves:
        A, b = leaf.get_polytope()
        if feasible(A, b):
            new_leaves.append(leaf)
    return new_leaves

################################## Top-n Tree ##################################

class RecTreeNode(object):
    '''
    A rec tree node describes regions of latent space in which a recommendation is made.
    It's value is the index of the item that will be recommended after all of its parent's values. 
    The corresponding part of latent space is the intersection of its region with all its parents regions.

    Attributes:
        - region: (A,b) describing polytopic region in H-representation. None for overall parent.
        - value: recommendation. None for overall parent.
        - parent: another RecTreeNode or None
        - children: (unused for now) list of RecTreeNodes
        - bound: coordinates of bounding ball (usually [0,1] for nonnegative or [-1,1])
    '''
    def __init__(self, region=None, value=None, parent=None, children=None, ball=None, tol=1e-3):
        self.parent = parent
        self.children = [] if children is None else children
        self.region = region
        self.value = value
        if parent is not None: 
            assert not (self.value is None), "value cannot be None if parent is not None!"
        
        # region bounding box for polytope creation
        if ball is None:
            if self.parent is not None:
                self.ball = self.parent.ball
            else:
                assert False, "must provide bounding ball for initial node!"
        else:
            self.ball = ball
        self.tol = tol
        

    def add_child(self, child):
        self.children.append(child)

    def get_region_list(self, tol=0):
        if self.parent is None:
            return []
        else:
            parent_region = self.parent.get_region_list()
            region = modify_polytope_away_from_origin(*self.region, self.tol) if self.tol > 0 else self.region
            return parent_region + [region]

    def get_rec_list(self):
        if self.parent is None:
            return []
        else:
            parent_recs = self.parent.get_rec_list()
            return parent_recs + [self.value]

    def get_polytope(self, ball=None):
        '''
        return a polytope representing the intersections of current region and 
        regions of all parents. 
        '''
        if self.region is None: return pc.Polytope()

        # ensures that polytopes are bounded
        A_ball, b_ball = ball if ball is not None else self.ball

        region = self.get_region_list()
        As = [A for A,_ in region]
        bs = [b for _,b in region]
        return (np.vstack([A_ball] + As),np.hstack([b_ball] + bs))

    def is_empty(self, ball=None):
        if self.region is None: return True, [np.nan]
        poly = self.get_polytope(ball=ball)
        start = time.time()
        empty = check_empty(poly, method='polytope-lp')
        time_fulldim = time.time() - start
        return empty, [time_fulldim]

def check_empty(poly, method='polytope-lp'):
    '''
    checks whether a polytope in (A,b) representation is empty 
    variety of methods 
    '''
    A, b = poly
    if method == 'polytope-fulldim':
        poly = pc.Polytope(A=A, b=b, normalize=False)
        empty = not pc.is_fulldim(poly)
    elif method == 'cvxpy':
        # this is 10x slower than polytope
        v = cvx.Variable(A.shape[1])
        prob = cvx.Problem(cvx.Minimize(cvx.norm(v)), [A@v <= b])
        try:
            prob.solve()
            empty = prob.status == "infeasible"
        except cvx.SolverError:
            empty = True
    elif method == 'polytope-lp':
        c = np.ones(A.shape[1])
        res = pc.solvers.lpsolve(c, A, b, solver='glpk') # 'mosek')
        empty = res['status'] != 0
    else:
        raise NotImplementedError('method {} not implemented'.format(method))

    return empty

################################## Visualization Stuff ##################################

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)

def plot_latent_space(H, leaves, nonnegative=False, ax=None, projection=[0,1], 
    color_by_top=0, title='Items in Latent Space', colorbar=True, figsize=None, arrowax=True):
    n_latent_features, n_items = H.shape
    cmap = discrete_cmap(n_items)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)

    # plotting polytope regions
    for leaf in leaves:
        A, b = leaf.get_polytope()
        A = A[:,projection] if n_latent_features > 2 else A
        plot_poly = pc.Polytope(A=A, b=b, normalize=False)
        corresponding_rec = leaf.get_rec_list()
        if pc.is_fulldim(plot_poly): 
            polyplot(plot_poly, ax=ax, color=cmap(corresponding_rec[color_by_top]), alpha=0.5)

    # plotting items
    im = ax.scatter(H[projection[0],:], H[projection[1],:], c=np.arange(n_items), cmap=cmap, 
                edgecolors='black', marker='o', s=50)
    im.set_clim([0,n_items])

    

    ax.axis('equal')
    xlim, ylim = np.amax(np.abs(H[projection]), axis=1)
    xmax = 1.3*xlim; ymax = 1.3*ylim
    if nonnegative:
        xmin = -0.1*xlim; ymin = -0.1*xlim
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
    else:
        xmin = -xmax; ymin = -ymax
        ax.set_xlim([-xmax,xmax])
        ax.set_ylim([-ymax,ymax])
    ax.margins(x=0,y=0)
    ax.set_title(title)

    if arrowax:
        # removing the default axis on all sides:
        for side in ['bottom','right','top','left']:
            ax.spines[side].set_visible(False)
        # removing the axis ticks
        plt.xticks([]) # labels 
        plt.yticks([])
        ax.xaxis.set_ticks_position('none') # tick markers
        ax.yaxis.set_ticks_position('none')

        # get width and height of axes object to compute 
        # matching arrowhead length and width
        dps = fig.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height
         
        # manual arrowhead width and length
        hw = 1./20.*(ymax-ymin) 
        hl = 1./20.*(xmax-xmin)
        lw = 2. # axis line width
        ohg = 0.3 # arrow overhang
         
        # compute matching arrowhead length and width
        yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
        yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
         
        # draw x and y axis
        ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
                 head_width=hw, head_length=hl, overhang = ohg, 
                 length_includes_head=False, clip_on = False) 
         
        ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
                 head_width=yhw, head_length=yhl, overhang = ohg, 
                 length_includes_head= False, clip_on = False) 
         

    if colorbar: plt.colorbar(im, ax=ax)
    return ax

def polyplot(poly, ax, color=None,
         hatch=None, alpha=1.0):
    if poly.dim != 2:
        raise Exception("Cannot plot polytopes of dimension larger than 2")
    if not pc.is_fulldim(poly):
        return None
    if color is None:
        color = np.random.rand(3)
    poly = _get_patch(
        poly, facecolor=color, hatch=hatch,
        alpha=alpha)
    ax.add_patch(poly)
    return ax


def _get_patch(poly1, **kwargs):
    """Return matplotlib patch for given Polytope.

    Example::

    > # Plot Polytope objects poly1 and poly2 in the same plot
    > import matplotlib.pyplot as plt
    > fig = plt.figure()
    > ax = fig.add_subplot(111)
    > p1 = _get_patch(poly1, color="blue")
    > p2 = _get_patch(poly2, color="yellow")
    > ax.add_patch(p1)
    > ax.add_patch(p2)
    > ax.set_xlim(xl, xu) # Optional: set axis max/min
    > ax.set_ylim(yl, yu)
    > plt.show()

    @type poly1: L{Polytope}
    @param kwargs: any keyword arguments valid for
        matplotlib.patches.Polygon
    """
    import matplotlib as mpl
    V = pc.extreme(poly1)
    rc, xc = pc.cheby_ball(poly1)
    x = V[:, 1] - xc[1]
    y = V[:, 0] - xc[0]
    mult = np.sqrt(x**2 + y**2)
    x = x / mult
    angle = np.arccos(x)
    corr = np.ones(y.size) - 2 * (y < 0)
    angle = angle * corr
    ind = np.argsort(angle)
    # create patch
    patch = mpl.patches.Polygon(V[ind, :], True, **kwargs)
    patch.set_zorder(0)
    return patch