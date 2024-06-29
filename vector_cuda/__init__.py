from typing import Union,List
import numpy as np
import cupy as cp


def is_cuda_available() -> bool:
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except cp.cuda.runtime.CUDARuntimeError: return False

def vector(components: List[Union[int,float]], device: str = "cpu") -> "Vector": return Vector(components,device)

def zeros(size: int,device: str = "cpu") -> "Vector": return Vector(components = [0]*size,device = device)

def ones(size: int,device: str = "cpu") -> "Vector": return Vector(components = [1]*size,device = device)

def rand(size: int,seed: Union[int,None] = None,device: str = "cpu") -> "Vector":
    if device == "cuda": cp.random.seed(seed)
    elif device == "cpu": np.random.seed(seed)
    if device.lower() == "cuda":
        if not is_cuda_available(): raise ValueError("CUDA is not available on this device")
        component = cp.random.rand(size).tolist()
    elif device.lower() == "cpu":
        component = np.random.rand(size).tolist()
    return Vector(component,device = device)

def zeros_like(vec: "Vector"): return zeros(vec.length,vec.device)

def ones_like(vec: "Vector"): return ones(vec.length,vec.device)

def rand_liek(vec: "Vector",seed: Union[int,None] = None): return rand(vec.length,seed,vec.device)


class Vector:
    def __init__(self, components: List[Union[int,float]], device: str = "cpu"):
        self.__device = device.lower()
        if self.__device == "cuda" and is_cuda_available(): self.__comp = cp.array(components)
        elif self.__device == "cpu": self.__comp = np.array(components)
        else: raise ValueError(f"{self.__device} not found on this device")

    @property
    def length(self) -> int: return len(self.__comp)

    @property
    def device(self) -> str: return self.__device

    def __getitem__(self,index): return self.__comp[index]
    
    def __setitem__(self,index,value): self.__comp[index] = value

    def __repr__(self) -> str: return f"<vector object at {hex(id(self))} size={self.length} device={self.__device}>"

    def array(self): return self.__comp

    def change_device(self,device: str):
        device = device.lower()
        if device == self.__device:
            return
        if device == "cuda":
            if not is_cuda_available(): raise ValueError("CUDA is not found on this device")
            self.__comp = cp.asarray(self.__comp)
            self.__device = "cuda"
        elif device == "cpu":
            self.__comp = cp.asnumpy(self.__comp)
            self.__device = "cpu"
        else: raise ValueError(f"unsupported device: {device}")
    
    def __add__(self,other: Union[int,float,"Vector"]) -> "Vector":
        if isinstance(other,(int,float)):
            if self.__device == "cuda": return Vector((self.__comp + other).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp + other).tolist(),device = "cpu")
        elif isinstance(other,Vector):
            if self.__device != other.__device: raise ValueError("both the vetors must be on the same device")
            if self.__device == "cuda": return Vector((self.__comp + other.__comp).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp + other.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for +: {type(other).__name__} with Vector")
 
    def __radd__(self,other: Union[int,float]) -> "Vector":
        if self.__device == "cuda": return Vector((other + self.__comp).tolist(),device = "cuda")
        if self.__device == "cpu": return Vector((other + self.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for +: {type(other).__name__} with Vector")

    def __sub__(self,other: Union[int,float,"Vector"]) -> "Vector":
        if isinstance(other,(int,float)):
            if self.__device == "cuda": return Vector((self.__comp - other).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp - other).tolist(),device = "cpu")
        elif isinstance(other,Vector):
            if self.__device != other.__device: raise ValueError("both the vectors must be on the same device")
            if self.__device == "cuda": return Vector((self.__comp + other.__comp).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp + other.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand typr for -: {type(other).__name__} with Vector")
    
    def __rsub__(self,other: Union[int,float]) -> "Vector":
        if self.__device == "cuda": return Vector((other - self.__comp).tolist(),device = "cuda")
        if self.__device == "cpu": return Vector((other - self.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for -: {type(other).__name__} with Vector")
    
    def __mul__(self,other: Union[int,float,"Vector"]) -> "Vector":
        if isinstance(other,(int,float)):
            if self.__device == "cuda": return Vector((self.__comp * other).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp * other).tolist(),device = "cpu")
        elif isinstance(other,Vector):
            if self.__device != other.__device: raise ValueError("both the vectors must be on thr same device")
            if self.__device == "cuda": return Vector((self.__comp * other.__comp).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp * other.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for *: {type(other).__name__} with Vector")
    
    def __rmul__(self,other: Union[int,float]) -> "Vector":
        if self.__device == "cuda": return Vector((other * self.__comp).tolist(),device = "cuda")
        elif self.__device == "cpu": return Vector((other * self.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for *: {type(other).__name__} with Vector")

    def __truediv__(self,other: Union[int,float,"Vector"]) -> "Vector":
        if isinstance(other,(int,float)):
            if self.__device == "cuda": return Vector((self.__comp / other).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp / other).tolist(),device = "cpu")
        elif isinstance(other,Vector):
            if self.__device != other.__device: raise ValueError("both the vectors must be on the same device")
            if self.__device == "cuda": return Vector((self.__comp / other.__comp).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp / other.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for /: {type(other).__name__} with Vector")
    
    def __rtruediv__(self,other: Union[int,float]) -> "Vector":
        if self.__device == "cuda": return Vector((other / self.__comp).tolist(),device = "cuda")
        elif self.__device == "cpu": return Vector((other / self.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for /: {type(other).__name__} with Vector")
    
    def __floordiv__(self,other: Union[int,float,"Vector"]) -> "Vector":
        if isinstance(other,(int,float)):
            if self.__device == "cuda": return Vector((self.__comp // other).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp // other).tolist(),device = "cpu")
            raise TypeError(f"unsupported operand type for //: {type(other).__name__} with Vector")
        elif isinstance(other,Vector):
            if self.__device != other.__device: raise ValueError("both the vectors must be on the same device")
            if self.__device == "cuda": return Vector((self.__comp // other.__comp).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp // other.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for //: {type(other).__name__} with Vector")

    def __rfloordiv__(self,other: Union[int,float]) -> "Vector":
        if self.__device == "cuda": return Vector((other // self.__comp).tolist(),device = "cuda")
        elif self.__device == "cpu": return Vector((other // self.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for //: {type(other).__name__} with Vector")

    def __matmul__(self,other: "Vector") -> int|float:
        if self.__device == "cuda": return cp.dot(self.__comp,other.__comp)
        elif self.__device == "cpu": return np.dot(self.__comp,other.__comp)
        raise TypeError(f"unsupported operand type for @: {type(other).__name__} with Vector")
    
    def __pow__(self,other: Union[int,float,"Vector"]) -> "Vector":
        if isinstance(other,(int,float)):
            if self.__device == "cuda": return Vector((self.__comp ** other).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp ** other).tolist(),device =  "cpu")
        elif isinstance(other,Vector):
            if self.__device != other.__device: raise ValueError("both the vectors must be on the same device")
            if self.__device == "cuda": return Vector((self.__comp ** other.__comp).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp ** other.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for **: {type(other).__name__} with Vector")
    
    def __rpow__(self,other: Union[int,float]) -> "Vector":
        if self.__device == "cuda": return Vector((other ** self.__comp).tolist(),device = "cuda")
        elif self.__device == "cpu": return Vector((other ** self.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for **: {type(other).__name__} with Vector")
    
    def __mod__(self,other: Union[int,float,"Vector"]) -> "Vector":
        if isinstance(other,(int,float)):
            if self.__device == "cuda": return Vector((self.__comp % other).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp % other).tolist(),device = "cpu")
        elif isinstance(other,Vector):
            if self.__device != other.__device: raise ValueError("both the vectors must be on the same device")
            if self.__device == "cuda": return Vector((self.__comp % other.__comp).tolist(),device = "cuda")
            elif self.__device == "cpu": return Vector((self.__comp % self.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for %: {type(other).__name__} with Vector")
        
    def __rmod__(self,other: Union[int,float]) -> "Vector":
        if self.__device == "cuda": return Vector((other % self.__comp).tolist(),device = "cuda")
        elif self.__device == "cpu": return Vector((other % self.__comp).tolist(),device = "cpu")
        raise TypeError(f"unsupported operand type for %: {type(other).__name__} with Vector")
    
    def __and__(self,other: Union[int,"Vector"]) -> int:
        if isinstance(other,int): return self.__comp & other
        elif isinstance(other,Vector): return self.__comp & other.__comp
        raise TypeError(f"unsupported operand type for &: {type(other).__name__} with Vector")
    
    def __rand__(self,other: int) -> int:
        if isinstance(other,int): return other & self.__comp
        raise TypeError(f"unsupported operand type for &: {type(other).__name__} with Vector")
    
    def __or__(self,other: Union[int,"Vector"]) -> int:
        if isinstance(other,int): return self.__comp or other
        elif isinstance(other,Vector): return self.__comp or other.__comp
        raise TypeError(f"unsupported operand type for |: {type(other).__name__} with Vector")
    
    def __ror__(self,other: int) -> int:
        if isinstance(other,int): return other | self.__comp
        raise TypeError(f"unsupported operand type for |: {type(other).__name__} with Vector")

    def __invert__(self) -> int: return ~self.__comp

    def __xor__(self,other: Union[int,"Vector"]) -> int:
        if isinstance(other,int): return self.__comp ^ other
        elif isinstance(other,Vector): return self.__comp ^ other.__comp
        raise TypeError(f"unsupported operand type for ^: {type(other).__name__} with Vector")

    def __rxor__(self,other: int) -> int:
        if isinstance(other,int): return other ^ self.__comp
        raise TypeError(f"unsupported operand type for ^: {type(other).__name__} with Vector")

    def __eq__(self,other: "Vector") -> bool: return self.norm() == other.norm()

    def __ne__(self,other: "Vector") -> bool: return self.norm() != other.norm()

    def __gt__(self,other: "Vector") -> bool: return self.norm() > other.norm()

    def __ge__(self,other: "Vector") -> bool: return self.norm() >= other.norm()

    def __lt__(self,other: "Vector") -> bool: return self.norm() < other.norm()

    def __le__(self,other: "Vector") -> bool: return self.norm() <= other.norm()
   
    def add(self,other: Union[int,float,"Vector"]) -> "Vector": return self + other

    def radd(self,other: Union[int,float,"Vector"]) -> "Vector": return other + other

    def sub(self,other: Union[int,float,"Vector"]) -> "Vector": return self - other

    def rsub(self,other: Union[int,float,"Vector"]) -> "Vector": return other - self

    def mul(self,other: Union[int,float,"Vector"]) -> "Vector": return self * other

    def rmul(self,other: Union[int,float,"Vector"]) -> "Vector": return other * self

    def matmul(self,other: Union[int,float,"Vector"]) -> "Vector": return self @ other

    def truediv(self,other: Union[int,float,"Vector"]) -> "Vector": return self / other

    def rtruediv(self,other: Union[int,float,"Vector"]) -> "Vector": return other / self

    def floordiv(self,other: Union[int,float,"Vector"]) -> "Vector": return self // other
    
    def rfloordiv(self,other: Union[int,float,"Vector"]) -> "Vector": return other // self

    def pow(self,other: Union[int,float,"Vector"]) -> "Vector": return self ** other

    def rpow(self,other: Union[int,float,"Vector"]) -> "Vector": return other ** self

    def mod(self,other: Union[int,float,"Vector"]) -> "Vector": return self % other

    def rmod(self,other: Union[int,float,"Vector"]) -> "Vector": return other % self

    def eq(self,other: "Vector") -> bool: return self == other

    def ne(self,other: "Vector") -> bool: return self != other

    def gt(self,other: "Vector") -> bool: return self > other

    def ge(self,other: "Vector") -> bool: return self >= other

    def lt(self,other: "Vector") -> bool: return self < other

    def le(self,other: "Vector") -> bool: return self <= other

    def scale(self,scalar) -> "Vector":
        if self.__device == "cuda": return Vector((self.__comp * scalar).tolist(),device = "cuda")
        elif self.__device == "cpu": return Vector((self.__comp * scalar).tolist(),device = "cpu")
        raise TypeError(f"{self.__device} invalid datatype for scaling, expected a Vector")

    def norm(self) -> Union[int,float]:
        if self.__device == "gpu": return cp.sqrt(self @ self)
        elif self.__device == "cpu": return np.sqrt(self @ self)

    def unit(self) -> "Vector":
        if self.__device == "cuda": return Vector((self.__comp / self.norm()).tolist(),device = "cuda")
        elif self.__device == "cpu": return Vector((self.__comp / self.norm()).tolist(),device = "cpu")

    def proj(self,on: "Vector") -> "Vector":
        if self.__device != on.__device: raise ValueError("both the vectors must be on the same device")
        return (self * on) / on.norm()

    def is_parallel(self,other: "Vector") -> bool:
        if self.__device != other.__device: raise ValueError("both the vectors must be on the same device")
        if self.__device == "cuda": return cp.allclose(self.__comp * other.__comp[::-1],self.__comp[::-1] * other.__comp)
        elif self.__device == "cpu": return np.allclose(self.__comp * other.__comp[::-1],self.__comp[::-1] * other.__comp)

    def is_othogonal(self,other: "Vector") -> bool:
        if self.__device != other.__device: raise ValueError("both the vectors must be on the same device")
        return self @ other == 0
