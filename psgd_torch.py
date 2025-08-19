# @title PSGD Shared Q code

"""
PSGD-Kron Newton/Whitening preconditioner with shared Kronecker factors for PyTorch.

This module extends the Kronecker product gradient/momentum whitening preconditioner (KronWhiten)
to share preconditioner factors Q across parameters with identical shapes, reducing memory usage
and potentially improving convergence through shared learning.

References:
- https://arxiv.org/abs/1512.04202
- https://arxiv.org/abs/2402.11858

Author: Xi-Lin Li, lixilinx@gmail.com (original)
Extended with shared factors implementation
"""

import opt_einsum
import torch
from collections import defaultdict


def norm_lower_bound_spd(A):
    """
    Returns a cheap lower bound for the spectral norm of a symmetric positive definite matrix A.
    """
    max_abs = torch.max(A.diagonal().real) # used to normalize A to avoid numerical under/over-flow
    if max_abs > 0:
        A = A/max_abs
        j = torch.argmax(torch.real(torch.sum(A * A.conj(), dim=1)))
        x = A[j] @ A
        return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A)
    else: # must have A=0
        return max_abs 
    

def lift2single(x):
    # lift half or lower precision to single precision; leave single or higher precision unchanged  
    return x.to(torch.float32) if torch.finfo(x.dtype).eps > 1e-6 else x


def init_kron(t, Scale=1.0, max_size=float("inf"), max_skew=1.0, dQ="QEQ"):
    """
    For a scalar or tensor t, we initialize its states (preconditioner Q and Lipschitz smoothness constant L), 
    and reusable contraction expressions for updating Q and preconditioning gradient.
    
    Returns [[Q, L], (expressions...)] where Q and L are the preconditioner factors and Lipschitz constants.
    """
    if dQ == "QUAD4P": # the only case that we fit P directly; so square Scale 
        Scale = Scale ** 2 
    shape = t.shape 
    if len(shape)==0: # scalar 
        Q = [Scale * torch.ones_like(t),]
        L = [lift2single(torch.zeros_like(t.real)),]
        exprA = opt_einsum.contract_expression(",->", Q[0].shape, t.shape)
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape) 
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape),]
        exprQs = [opt_einsum.contract_expression(",->", Q[0].shape, t.shape),]
    else: # tensor 
        if len(shape) > 26:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; einsum runs out of letters; replace 26 with larger numbers.")   
            
        scale = Scale ** (1/len(shape)) 
    
        Q, L = [], []
        exprGs, exprQs = [], []
        piece1A, piece2A, piece3A = [], "", "" # used for getting the subscripts for exprA
        piece1P, piece2P, piece3P, piece4P = [], [], "", "" # used for getting the subscripts for exprP
        for i, size in enumerate(shape):
            L.append(lift2single(torch.zeros([], dtype=t.real.dtype, device=t.device)))
            if size <= 1 or size > max_size or size**2 > max_skew * t.numel():
                # use diagonal matrix as preconditioner for this dim 
                Q.append(scale * torch.ones(size, dtype=t.dtype, device=t.device))
                
                piece1A.append(opt_einsum.get_symbol(i))
                piece2A = piece2A + opt_einsum.get_symbol(i)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                piece1P.append(opt_einsum.get_symbol(i + 26))
                piece2P.append(opt_einsum.get_symbol(i + 26))
                piece3P = piece3P + opt_einsum.get_symbol(i + 26)
                piece4P = piece4P + opt_einsum.get_symbol(i + 26)
                
                piece1 = "".join([opt_einsum.get_symbol(i+26) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                subscripts = piece1 + "," + piece1 + "->" + opt_einsum.get_symbol(i+26)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

                subscripts = opt_einsum.get_symbol(i+26) + "," + piece1 + "->" + piece1
                exprQs.append(opt_einsum.contract_expression(subscripts, Q[-1].shape, t.shape))
            else: # use matrix preconditioner for this dim 
                Q.append(scale * torch.eye(size, dtype=t.dtype, device=t.device))

                piece1A.append(opt_einsum.get_symbol(i) + opt_einsum.get_symbol(i + 26))
                piece2A = piece2A + opt_einsum.get_symbol(i + 26)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                a, b, c = opt_einsum.get_symbol(i), opt_einsum.get_symbol(i + 26), opt_einsum.get_symbol(i + 805)
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b
                
                piece1 = "".join([opt_einsum.get_symbol(i+26) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                piece2 = "".join([opt_einsum.get_symbol(i+805) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                subscripts = piece1 + "," + piece2 + "->" + opt_einsum.get_symbol(i+26) + opt_einsum.get_symbol(i+805)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

                subscripts = opt_einsum.get_symbol(i+26) + opt_einsum.get_symbol(i+805) + "," + piece2 + "->" + piece1
                exprQs.append(opt_einsum.contract_expression(subscripts, Q[-1].shape, t.shape))
        
        subscripts = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprA = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], t.shape)

        subscripts = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        exprP = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape)
    
    exprGs, exprQs = tuple(exprGs), tuple(exprQs)
    if dQ == "QEP": 
        return [[Q, L], (exprP, exprGs, exprQs)]
    elif dQ == "EQ": 
        return [[Q, L], (exprP, exprGs, exprA)]
    elif (dQ == "QEQ") or (dQ == "QUAD"):
        return [[Q, L], (exprP, exprGs)]
    else: # the only case that we fit P directly 
        assert dQ == "QUAD4P", "Invalid choice for dQ" 
        return [[Q, L], (exprA, exprGs)]


def balance_kron_precond(Q):
    """
    Balance the dynamic ranges of the factors of Q to avoid over/under-flow.
    """
    order = len(Q)  # order of tensor or the number of factors in Q 
    if order>1:
        norms = [torch.max(torch.abs(q)) for q in Q]
        gmean = torch.prod(torch.stack(norms))**(1/order) # geometric mean 
        for i, q in enumerate(Q):
            q.mul_(gmean/norms[i]) 


def update_precond_kron_eq(QL, exprs, V, Hvp, lr=0.1, betaL=0.9):
    """
    The raw function for updating the Kron preconditioner Q and Lipschitz smoothness constant L with pair (V, Hvp),
    where Q is update as dQ = E*Q, 
    the pair (V, Hvp) can be (vector, hess-vector-prod) or (randn, gradient/momentum).  
    The damping logic is not included here. 
    """
    Q, L = QL
    _, exprGs, exprA = exprs
        
    def solve_triangular_right(X, A):
        # return X @ inv(A)
        if X.dim()>1: 
            return torch.linalg.solve_triangular(A, X, upper=True, left=False)
        else: # torch.linalg.solve_triangular complains if X.dim() < 2. So insert None.
            return torch.linalg.solve_triangular(A, X[None,:], upper=True, left=False)[0]     
    
    A = exprA(*Q, Hvp)

    order = V.dim()
    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1]) # permute dims like [0,1,2,3,4] -> [1,2,3,4,0]
    for i, q in enumerate(Q):
        conjB = conjB/q if q.dim()<2 else solve_triangular_right(conjB, q)
        if i < order - 1: # transpose dims like [1,2,3,4,0]->[0,2,3,4,1]->[0,1,3,4,2]->[0,1,2,4,3]->[0,1,2,3,4]
            conjB = torch.transpose(conjB, i, order - 1) 

    for i, q in enumerate(Q):
        term1 = exprGs[i](A, A.conj())
        term2 = exprGs[i](conjB.conj(), conjB)
                   
        if q.dim() < 2: # q is a diagonal matrix or scalar preconditioner
            ell = torch.max(torch.real(term1 + term2))
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))      
        else: # q is a matrix preconditioner 
            ell = norm_lower_bound_spd(term1 + term2)
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * torch.triu(term1 - term2) @ q)

    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def precond_grad_kron(QL, exprs, G):
    """
    Precondition gradient G with Kron preconditioner Q. 
    """
    Q, exprP = QL[0], exprs[0]
    return exprP(*[q.conj() for q in Q], *Q, G) 


def update_precond_kron_whiten_eq(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q as dQ = E*Q.
    """
    V = torch.randn_like(G)
    update_precond_kron_eq(QL, exprs, V, G + damping*V, lr=lr, betaL=betaL)
    

def update_precond_kron_whiten_qep(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q as dQ = Q*E*P. 
    """   
    Q, L = QL
    exprP, exprGs, exprQs = exprs
    
    # balancing is not optional as L for each factor is not scaling invariant 
    balance_kron_precond(Q) 

    total_numel = G.numel() 
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*torch.randn_like(G)) 
    for i, q in enumerate(Q):
        QPg = exprQs[i](q, Pg)
        term1 = exprGs[i](QPg, QPg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() * q * q.conj()
            ell = torch.max(torch.real(term1 + term2)) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            term2 = total_numel/q.shape[0] * q @ q.H
            ell = norm_lower_bound_spd(term1 + term2)
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * (term1 - term2) @ q)


def update_precond_kron_whiten_qeq(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q as dQ = Q*E*Q. 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    
    total_numel = G.numel() 
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*torch.randn_like(G)) 
    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() # times I
            ell = torch.max(torch.real(term1)) + term2 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            term2 = total_numel/q.shape[0] # times I
            ell = norm_lower_bound_spd(term1) + term2
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * (q @ term1 - q * term2))
            
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def update_precond_kron_whiten_quad(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q with a quadratic form. 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    
    total_numel = G.numel() 
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*torch.randn_like(G))   
    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() # times I
            ell = torch.max(torch.real(term1)) + term2 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            gain = 1 - lr/2/L[i] * (term1 - term2)
            q.mul_(gain * gain) 
        else: # matrix Q
            term2 = total_numel/q.shape[0] # times I
            ell = norm_lower_bound_spd(term1) + term2
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            p = q - lr/2/L[i] * (term1 @ q - term2 * q) 
            p = p - lr/2/L[i] * (p @ term1 - p * term2) 
            q.data = (p + p.H)/2 # p must be symmetric/hermitian  
            
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def update_precond_kron_whiten_quad4p(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Almost the same as function update_precond_kron_whiten_quad except that fitting P directly. 
    This is the only case that we fit P directly (Q here is P). Vulnerable to numerical errors.  
    """   
    Q, L = QL
    exprA, exprGs = exprs

    total_numel = G.numel() 
    Pg = exprA(*Q, G + damping*torch.randn_like(G)) # Q actually is P; so just applying all its factors once.
    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() # times I
            ell = torch.max(torch.real(term1)) + term2 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            gain = 1 - lr/L[i] * (term1 - term2)
            q.mul_(gain * gain) 
        else: # matrix Q
            term2 = total_numel/q.shape[0] # times I
            ell = norm_lower_bound_spd(term1) + term2
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            p = q - lr/L[i] * (term1 @ q - term2 * q) 
            p = p - lr/L[i] * (p @ term1 - p * term2) 
            q.data = (p + p.H)/2 # p must be symmetric/hermitian  
            
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


class KronWhitenShared:
    """
    Implements the PSGD optimizer with shared Kronecker product gradient/momentum whitening preconditioners.
    
    This version shares preconditioner factors Q across parameters with identical shapes, which:
    - Reduces memory usage by storing Q only once per unique shape
    - Potentially improves convergence through shared learning across similar layers
    - Updates shared Q with accumulated gradient information from all parameters sharing that shape
    
    Key differences from standard KronWhiten:
    - Groups parameters by shape and shares Q factors
    - Accumulates gradient information before updating shared Q
    - Manages shared state across parameter groups
    """
    def __init__(self,  params_with_grad, 
                 preconditioner_max_size=float("inf"), preconditioner_max_skew=1.0, preconditioner_init_scale:float|None=None,
                 lr_params=0.001, lr_preconditioner=0.1, betaL=0.9, damping=1e-9, momentum=0.0,
                 grad_clip_max_amp=float("inf"), preconditioner_update_probability=1.0, whiten_grad=True, dQ="QEQ",
                 share_factors=True):
        """
        Args:
            share_factors: If True, shares Q factors across parameters with identical shapes.
                          If False, behaves like standard KronWhiten.
        """
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner 
        self.betaL = betaL
        self.damping = damping
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_amp = grad_clip_max_amp
        self.preconditioner_update_probability = preconditioner_update_probability
        
        # protected members
        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad]
        self._num_params = sum([p.numel() for p in self._params_with_grad])
        self._whiten_grad = whiten_grad
        if not whiten_grad:
            assert self.momentum > 0, "Cannot whiten momentum if the momentum setting is zero."
        self._dQ = dQ
        self._share_factors = share_factors
        
        # Initialize shared or per-parameter states
        if share_factors:
            self._init_shared_states(preconditioner_init_scale)
        else:
            self._init_individual_states(preconditioner_init_scale)
            
        # Momentum buffers
        self._ms, self._counter_m = None, 0
        
        # Select update function based on dQ
        if dQ == "QUAD4P":
            assert max([torch.finfo(p.dtype).eps for p in self._params_with_grad]) < 1e-6, "Directly fitting P needs at least single precision"
            self._update_precond = update_precond_kron_whiten_quad4p
            self._precond_grad = lambda QL, exprs, G: exprs[0](*QL[0], G)
        else:
            self._precond_grad = precond_grad_kron            
            if dQ == "QEP":
                self._update_precond = update_precond_kron_whiten_qep
            elif dQ == "EQ":
                self._update_precond = update_precond_kron_whiten_eq
            elif dQ == "QEQ":
                self._update_precond = update_precond_kron_whiten_qeq
            elif dQ == "QUAD":
                self._update_precond = update_precond_kron_whiten_quad
                
    def _init_shared_states(self, preconditioner_init_scale):
        """Initialize shared states grouped by parameter shapes."""
        print("\n" + "="*60)
        print("INITIALIZING SHARED KRONECKER FACTORS")
        print("="*60)
        
        # Group parameters by shape
        self._shape_groups = defaultdict(list)
        for idx, param in enumerate(self._params_with_grad):
            shape_key = tuple(param.squeeze().shape)
            self._shape_groups[shape_key].append(idx)
        
        # Create shared Q, L, and expressions for each unique shape
        self._shared_QLs_exprs = {}
        self._param_to_shape = {}
        
        # Track which shapes are new vs reused
        shape_counter = 0
        param_counter = 0
        
        for shape_key, param_indices in self._shape_groups.items():
            shape_counter += 1
            
            # First parameter with this shape - CREATE NEW Q
            first_idx = param_indices[0]
            param_counter += 1
            print(f"\n[NEW Q #{shape_counter}] Creating new Q factors for shape {shape_key}")
            print(f"  └─ Parameter {first_idx}: shape {shape_key} -> CREATING new Q")
            
            # Additional parameters with same shape - REUSE Q
            for idx in param_indices[1:]:
                param_counter += 1
                print(f"  └─ Parameter {idx}: shape {shape_key} -> REUSING Q #{shape_counter}")
            
            # Use first param of this shape to initialize
            first_param = self._params_with_grad[first_idx].squeeze()
            
            if preconditioner_init_scale is None:
                # Will initialize on the fly
                self._shared_QLs_exprs[shape_key] = None
            else:
                QL_exprs = init_kron(first_param, preconditioner_init_scale, 
                                    self._preconditioner_max_size, self._preconditioner_max_skew, self._dQ)
                self._shared_QLs_exprs[shape_key] = QL_exprs
            
            # Map each parameter index to its shape key
            for idx in param_indices:
                self._param_to_shape[idx] = shape_key
        
        # Summary statistics
        print("\n" + "-"*60)
        print("SUMMARY:")
        print(f"  Total parameters: {param_counter}")
        print(f"  Unique Q factors created: {len(self._shape_groups)}")
        print(f"  Memory savings: {param_counter - len(self._shape_groups)} Q factors saved")
        
        if len(self._shape_groups) < param_counter:
            savings_pct = (1 - len(self._shape_groups)/param_counter) * 100
            print(f"  Efficiency gain: {savings_pct:.1f}% reduction in Q storage")
        
        print("-"*60)
                
        if preconditioner_init_scale is None:
            print(f"\nNOTE: Preconditioner scale will be set on the fly during first step.")
                
    def _init_individual_states(self, preconditioner_init_scale):
        """Initialize individual states for each parameter (standard KronWhiten behavior)."""
        if preconditioner_init_scale is None:
            self._QLs_exprs = None
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._QLs_exprs = [init_kron(p.squeeze(), preconditioner_init_scale, 
                                       self._preconditioner_max_size, self._preconditioner_max_skew, self._dQ) 
                             for p in self._params_with_grad]

    @torch.no_grad()
    def step(self, closure):
        """
        Performs one step of PSGD with shared Kronecker product gradient/momentum whitening preconditioner.
        """
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = [g.squeeze() for g in torch.autograd.grad(loss, self._params_with_grad)]
        
        if self._share_factors:
            return self._step_shared(grads, closure_returns)
        else:
            return self._step_individual(grads, closure_returns)
    
    def _step_shared(self, grads, closure_returns):
        """Step with shared preconditioner factors."""
        # Initialize shared factors on the fly if needed
        for shape_key in self._shape_groups.keys():
            if self._shared_QLs_exprs[shape_key] is None:
                # Use average statistics from all params of this shape for initialization
                param_indices = self._shape_groups[shape_key]
                shape_grads = [grads[idx] for idx in param_indices]
                avg_scale = torch.mean(torch.stack([(torch.mean((torch.abs(g))**4))**(-1/8) for g in shape_grads]))
                
                first_grad = shape_grads[0]
                self._shared_QLs_exprs[shape_key] = init_kron(first_grad, avg_scale,
                                                             self._preconditioner_max_size, 
                                                             self._preconditioner_max_skew, self._dQ)
        
        # Update momentum
        if self.momentum > 0:
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
            [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, grads)]
        else:
            self._ms, self._counter_m = None, 0
        
        # Update shared preconditioners
        if torch.rand([]) < self.preconditioner_update_probability:
            for shape_key, param_indices in self._shape_groups.items():
                QL_exprs = self._shared_QLs_exprs[shape_key]
                
                # Accumulate gradient information from all params sharing this shape
                if self._whiten_grad:
                    accumulated_grads = [grads[idx] for idx in param_indices]
                else:
                    accumulated_grads = [self._ms[idx] for idx in param_indices]
                
                # Update shared Q with accumulated information
                # We can either average the gradients or update with each one sequentially
                # Here we'll update with averaged gradient for stability
                if len(accumulated_grads) > 1:
                    avg_grad = torch.mean(torch.stack(accumulated_grads), dim=0)
                else:
                    avg_grad = accumulated_grads[0]
                    
                self._update_precond(*QL_exprs, avg_grad, lr=self.lr_preconditioner, 
                                   betaL=self.betaL, damping=self.damping)
        
        # Precondition gradients using shared factors
        pre_grads = []
        for idx, g in enumerate(grads):
            shape_key = self._param_to_shape[idx]
            QL_exprs = self._shared_QLs_exprs[shape_key]
            
            if self.momentum > 0:
                pre_grads.append(self._precond_grad(*QL_exprs, self._ms[idx]))
            else:
                pre_grads.append(self._precond_grad(*QL_exprs, g))
        
        # Gradient clipping
        lr = self.lr_params
        if self.grad_clip_max_amp < float("inf"):
            avg_amp = torch.sqrt(torch.real(sum([torch.sum(g*g.conj()) for g in pre_grads]))/self._num_params)
            if avg_amp > self.grad_clip_max_amp:
                lr = lr * self.grad_clip_max_amp / avg_amp
        
        # Update parameters
        [param.subtract_(lr*g.view_as(param)) for (param, g) in zip(self._params_with_grad, pre_grads)]
        
        return closure_returns
    
    def _step_individual(self, grads, closure_returns):
        """Step with individual preconditioner factors (standard KronWhiten behavior)."""
        # Initialize on the fly if needed
        if self._QLs_exprs is None:
            self._QLs_exprs = [init_kron(g, (torch.mean((torch.abs(g))**4))**(-1/8), 
                                       self._preconditioner_max_size, self._preconditioner_max_skew, self._dQ) 
                             for g in grads]
        
        # Update momentum
        if self.momentum > 0:
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
            [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, grads)]
        else:
            self._ms, self._counter_m = None, 0
        
        # Update preconditioners
        if torch.rand([]) < self.preconditioner_update_probability:
            if self._whiten_grad:
                [self._update_precond(*QL_exprs, g, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping) 
                 for (QL_exprs, g) in zip(self._QLs_exprs, grads)]
            else:
                [self._update_precond(*QL_exprs, m, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping) 
                 for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
        
        # Precondition gradients
        if self.momentum > 0:
            pre_grads = [self._precond_grad(*QL_exprs, m) for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
        else:
            pre_grads = [self._precond_grad(*QL_exprs, g) for (QL_exprs, g) in zip(self._QLs_exprs, grads)]
        
        # Gradient clipping
        lr = self.lr_params
        if self.grad_clip_max_amp < float("inf"):
            avg_amp = torch.sqrt(torch.real(sum([torch.sum(g*g.conj()) for g in pre_grads]))/self._num_params)
            if avg_amp > self.grad_clip_max_amp:
                lr = lr * self.grad_clip_max_amp / avg_amp
        
        # Update parameters
        [param.subtract_(lr*g.view_as(param)) for (param, g) in zip(self._params_with_grad, pre_grads)]
        
        return closure_returns
