if __name__ == '__main__':
    file_paths = [
        '.....' # put your paths to fits data here
    ]
    save_path = '.....' # add your path to save results
    os.makedirs(save_path, exist_ok=True)

    # get fits data
    file_path = '... .fits' # add your path
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        teff = header.get('TEFF', None)       # effective star's temperature
        radius = header.get('RADIUS', None)   # star's radius

    processor = DataProcessor(file_paths, save_path)
    processor.process_data()
    print("flare analisys completed.")

# unittests
class TestFlareAnalysis(unittest.TestCase):
    def test_calculate_flare_amplitude(self):
        time_series = pd.DataFrame({'Norm.Flux': [1.0, 1.2, 1.5, 1.3, 1.1]})
        flare_indices = [2]
        gp_curve_interp = np.array([1.0, 1.1, 1.2, 1.3, 1.1])
        amplitudes = FlareAnalysis.calculate_flare_amplitude(time_series, flare_indices, gp_curve_interp)
        self.assertEqual(amplitudes, [0.3])

    def test_calculate_flare_duration(self):
        time_series = pd.DataFrame({'TIME': [1, 2, 3, 4, 5, 6, 7], 'Norm.Flux': [1.0, 1.2, 1.5, 1.3, 1.1, 1.0, 0.9]})
        flare_index = 2
        duration = FlareAnalysis.calculate_flare_duration([flare_index], time_series)
        self.assertAlmostEqual(duration[0], 1 / (24 * 3600), places=4)

    def test_identify_flares(self):
        time_series = pd.DataFrame({'TIME': [1, 2, 3, 4, 5], 'Norm.Flux': [1.0, 1.2, 1.5, 1.3, 1.1]})
        gp_curve = np.array([1.0, 1.1, 1.2, 1.3, 1.1])
        flare_indices, _, _ = FlareAnalysis.identify_flares(time_series, gp_curve)
        self.assertEqual(flare_indices, np.array([2]))

    def test_calculate_flare_energy(self):
        time_series = pd.DataFrame({'Norm.Flux': [1.0, 1.2, 1.5, 1.3, 1.1]})
        flare_indices = [2]
        teff = 5778
        radius = 0.865
        gp_curve_interp = np.array([1.0, 1.1, 1.2, 1.3, 1.1])
        flare_energy = FlareAnalysis.calculate_flare_energy(time_series, flare_indices, teff, radius, gp_curve_interp)
        self.assertGreater(flare_energy[0], 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
