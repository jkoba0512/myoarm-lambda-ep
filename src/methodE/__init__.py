"""
Method E: 神経系コンポーネント

- LIFProprioceptor  : LIF ニューロンによる固有受容器（関節角・速度の符号化）
- IzhikevichReflexArc : Izhikevich Fast-Spiking ニューロンによる反射弓
"""

from methodE.lif_proprioceptor import LIFProprioceptor
from methodE.izhikevich_reflex import IzhikevichReflexArc

__all__ = ["LIFProprioceptor", "IzhikevichReflexArc"]
