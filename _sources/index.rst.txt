.. |logo| image:: _static/myoverse_logo.png
   :height: 80px
   :align: middle

Welcome to |logo|
===========================

**The AI toolkit for myocontrol research**

MyoVerse is your cutting-edge **research** companion for unlocking the secrets hidden within biomechanical data! It's specifically designed for exploring the complex interplay between **electromyography (EMG)** signals, **kinematics** (movement), and **kinetics** (forces).

Leveraging the power of **PyTorch** and **PyTorch Lightning**, MyoVerse provides a comprehensive suite of tools for researchers and developers working with myoelectric signal analysis and AI-driven biomechanical applications.

.. raw:: html

    <div>
        <form class="bd-search align-items-center" action="search.html" method="get">
          <input type="search" class="form-control search-front-page" name="q" id="search-input" placeholder="&#128269; Search the docs ..." aria-label="Search the docs ..." autocomplete="off">
        </form>
    </div>

Key Features
-----------

* **Data loaders** and **preprocessing filters** tailored for biomechanical signals
* Peer-reviewed **AI models** and components for analysis and prediction tasks
* Comprehensive visualization tools
* Essential **utilities** to streamline the research workflow

.. important::
   MyoVerse is built for **research**. While powerful, it's evolving and may not have the same level of stability as foundational libraries like NumPy.

Package Structure
----------------

* **myoverse**: Main package containing:
   * **datasets**: Data loaders, dataset creators, and preprocessing filters
   * **models**: AI models and components for training and evaluation
   * **utils**: Support for data handling, model training, and analysis
* **examples**: Practical examples including tutorials and use cases

Research
----------------------

MyoVerse has been used in several publications:

* IEEE Transactions on Biomedical Engineering (10.1109/TBME.2024.3432800)
* Journal of Neural Engineering (10.1088/1741-2552/ad3498)
* IEEE Transactions on Neural Systems and Rehabilitation Engineering (10.1109/TNSRE.2023.3295060)
* And more...

.. raw:: html

    <!-- Raw HTML for Carousel -->
    <style>
    .paper-carousel-container {
        width: 95%; /* Use more width */
        max-width: 80em; /* Increased max width */
        height: 700px; /* Match image max-height */
        overflow: hidden;
        margin: 25px auto; /* Slightly more margin */
        border: 0px solid #ddd;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .paper-carousel-track {
        display: flex;
        /* Adjusted calculation: (max-width + margin) * num images * 2 */
        /* (650px + 15px) * 6 * 2 = 7980px */
        width: 7980px; /* Use fixed pixel value based on max-width */
        animation: scroll 60s linear infinite; /* Slower scroll for bigger images */
    }

    .paper-carousel-track a {
        display: block; /* Make anchor fill space */
        margin-right: 15px; /* Space between images */
        flex-shrink: 0; /* Prevent shrinking */
    }
    
    .paper-carousel-track img {
        width: auto; /* Auto width based on height */
        max-width: 700px; /* Add max-width */
        height: auto;
        max-height: 700px; /* Increased max height */
        object-fit: contain;
        display: block; /* Remove potential extra space below image */
        border: 1px solid #eee; /* Optional subtle border for each image */
    }
    
    @keyframes scroll {
        0% {
            transform: translateX(0);
        }
        100% {
            /* Adjusted translation: -(max-width + margin) * num images */
            /* -(650px + 15px) * 6 = -3990px */
            transform: translateX(-3990px); /* Use fixed pixel value */
        }
    }
    
    .paper-carousel-container:hover .paper-carousel-track {
        animation-play-state: paused;
    }
    </style>
    
    <div class="paper-carousel-container">
        <div class="paper-carousel-track">
            <!-- List images twice, wrapped in links -->
            <a href="https://doi.org/10.1109/EMBC48229.2022.9870937" target="_blank"><img src="_static/papers/Accurate.jpg" alt="Accurate Paper"></a>
            <a href="https://doi.org/10.1109/TNSRE.2023.3295060" target="_blank"><img src="_static/papers/Proportional.jpg" alt="Proportional Paper"></a> 
            <a href="https://doi.org/10.1088/1741-2552/ad3498" target="_blank"><img src="_static/papers/Influence.jpg" alt="Influence Paper"></a>
            <a href="http://www.iadisportal.org/ijcsis/papers/2024190101.pdf" target="_blank"><img src="_static/papers/Analysis.jpg" alt="Analysis Paper"></a>          
            <a href="https://doi.org/10.1109/TBME.2024.3432800" target="_blank"><img src="_static/papers/Learning.jpg" alt="Learning Paper"></a>
            <a href="https://doi.org/10.1109/TNSRE.2024.3472063" target="_blank"><img src="_static/papers/Identification.jpg" alt="Identification Paper"></a>
            <!-- Add a new row to create a continuous scroll -->
            <a href="https://doi.org/10.1109/EMBC48229.2022.9870937" target="_blank"><img src="_static/papers/Accurate.jpg" alt="Accurate Paper"></a>
            <a href="https://doi.org/10.1109/TNSRE.2023.3295060" target="_blank"><img src="_static/papers/Proportional.jpg" alt="Proportional Paper"></a> 
            <a href="https://doi.org/10.1088/1741-2552/ad3498" target="_blank"><img src="_static/papers/Influence.jpg" alt="Influence Paper"></a>
            <a href="http://www.iadisportal.org/ijcsis/papers/2024190101.pdf" target="_blank"><img src="_static/papers/Analysis.jpg" alt="Analysis Paper"></a>          
            <a href="https://doi.org/10.1109/TBME.2024.3432800" target="_blank"><img src="_static/papers/Learning.jpg" alt="Learning Paper"></a>
            <a href="https://doi.org/10.1109/TNSRE.2024.3472063" target="_blank"><img src="_static/papers/Identification.jpg" alt="Identification Paper"></a>
             
        </div>
    </div>
    <!-- End Raw HTML -->

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   auto_examples/index.rst
   api_documentation.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing.rst
   Changelog <../CHANGELOG.md>
