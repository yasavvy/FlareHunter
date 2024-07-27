## FlareHunter: Stellar Flare Identification & Analysis

FlareHunter is designed for robust and efficient analysis of stellar flares detected in data from the Transiting Exoplanet Survey Satellite (TESS). Built with SOLID principles in mind, FlareHunter provides a powerful and flexible tool for astrophysical research.

Key Features:

* Automated Threshold Selection: FlareHunter utilizes cross-validation to determine the optimal threshold for flare identification, ensuring reliable and objective results.

* Comprehensive Unit Tests: The code includes extensive unit tests for the FlareAnalysis class, guaranteeing accuracy and robustness.

* Clear Structure and Documentation: The code is well-organized with comments explaining logic and key aspects, making it easy to understand and adapt.

* Leveraging Standard Libraries: FlareHunter relies on standard Python libraries like numpy, pandas, scipy, matplotlib, and george, making it accessible and readily usable by other developers.

# Installation

Install Required Libraries:
pip install numpy pandas scipy matplotlib george astropy scikit-learn
(Or use requirements.txt)

Load Data: use your paths to FITS files.
Run FlareHunter: Execute the flare_analysis.py script.
Results: Flare data will be saved to the flares_data.txt file in your save_path folder.

# Example Usage

See the usage_example.py for details

# Process data
processor = DataProcessor(file_paths, save_path)
processor.process_data()
print("Flare data processing is complete and saved to file.")

# Unit Tests

The repository includes unit tests for the FlareAnalysis class. Run them with the command: python -m unittest flare_analysis.py.

# License

This project is licensed under the MIT License

# Author

Mrs. Elena Savvina

# Contributing

Contributions are welcome! Please read the CONTRIBUTING.md file before submitting a pull request.
