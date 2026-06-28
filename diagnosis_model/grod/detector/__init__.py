"""Vendored RF-DETR decoder core (model + criterion + matcher + ops +
backbone wrapper + config), copied from the fork at base 1.6.5.post0 so the GROD
path can build/run the detector without importing the rfdetr package. Pure
PyTorch, no compiled kernels. GROD heads / backbone variants stay in
diagnosis_model/grod/{heads,backbone} and register via the seam modules here."""
