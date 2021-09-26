.. _contrib:


Contribution
====================



Development Setup
----------------------------

1. Fork and clone the forked repository
2. Create a virtual env (optional)
3. [Install and Setup Poetry](https://python-poetry.org/docs/#installation)
4. Install package dependencies [using poetry](https://python-poetry.org/docs/cli/#install) or set them up manually
5. Start development


Scripts
------------------------

To create documentation for the `./tf_al` directory. Execute following command
in `./docs`

.. code-block:: shell

    $ poetry run make html


To clear the generated documentation use following command.


.. code-block:: shell

    $ poetry run make clean


To perform automated unittests run following command in the root package directory.

.. code-block:: shell

    $ poetry run pytest


To generate additional coverage reports run.

.. code-block:: shell

    $ pytest --cov