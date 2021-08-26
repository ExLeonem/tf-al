import pytest
import numpy as np
from tf_al.query import AcquisitionFunction, name


class TestAcquisitionFunction:


    def test_unlimited_kwargs(self):
        first = 1
        second = 5
        fn = AcquisitionFunction(first=first, second=second)
        assert fn.first == first and fn.second == second


    def test_unlimited_kwargs_invalid(self):
        with pytest.raises(ValueError):
            fn = AcquisitionFunction(select_min="hello")


    def test_name_decorator(self):
        fn_name = "Mock function"
        @name(fn_name)
        class MockAcq(AcquisitionFunction):

            def do_something(self):
                pass

        obj = MockAcq()
        assert obj.get_fn_name() == fn_name

    
    def test_name_decorator_invalid(self):

        with pytest.raises(ValueError):
            @name
            class MockAcq(AcquisitionFunction):
                def do_something(self):
                    pass


            
        