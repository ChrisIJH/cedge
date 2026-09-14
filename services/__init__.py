"""Not a shared library — each service is built and deployed independently
(see its own Dockerfile). This package marker exists only so pytest can
import services.<name>.app by qualified name in tests."""