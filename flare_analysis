import numpy as np
import os
from scipy.signal import find_peaks, butter, lfilter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import george
from george import kernels
from astropy.io import fits
from scipy.integrate import quad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import unittest

# Constants
class Constants:
    H_CGS = 6.626e-27  # Planck constant (erg*s)
    C_CGS = 3.0e10     # speed of light (cm/s)
    K_CGS = 1.381e-16  # Boltzmann constant (erg/K)
    LAMBDA_MIN = 600e-7  # 600 nm in cm
    LAMBDA_MAX = 1000e-7  # 1000 nm in cm
    TIME_FACTOR = 120  # sampling rate every 2 minutes (120 s)
    SIG = 5.6704e-5  # Stefan-Boltzmann constant

# Filters
class Filters:
    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        """
        Function for implementing a Butterworth bandbass filter
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

# Gaussian models
class GaussianProcessModel:
    def __init__(self, skip_points=1):
        self.skip_points = skip_points

    def fit_and_predict(self, time_series, plot=False):
        """
        trains a GP model and predicts values
        """
        # every skip_points-th point is used
        time_series = time_series.iloc[::self.skip_points]

        X = time_series['TIME'].values.reshape(-1, 1)  # time
        Y = time_series['Norm.Flux'].values  # normalized flux

        # create & train GP model
        kernel = 1.0 * kernels.ExpSquaredKernel(1.0)
        gp = george.GP(kernel)
        gp.compute(X)
        # predict values
        X_pred = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
        Y_pred, _ = gp.predict(Y, X_pred)

        if not plot:
            # interpolation to match the dimension of the original array
            interp_func = interp1d(X_pred.flatten(), Y_pred.flatten(), kind='linear', fill_value='extrapolate')
            Y_pred_interp = interp_func(X.flatten())
            return Y_pred_interp

        # visualisation using matplotlib
        plt.figure(figsize=(10, 6))
        plt.scatter(time_series['TIME'], time_series['Norm.Flux'], c='b', label='Original Flux', s=2)
        plt.plot(X_pred, Y_pred, color='red', label='GP Fit', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Norm.Flux')
        plt.title('Original Data with GP Fit')
        plt.legend()
        plt.show()

        return Y_pred.flatten()

# Calculations
class PhysicalCalculations:
    @staticmethod
    def planck_function(wavelength, temperature):
        """
        Planck function according to the CGS.
        """
        return (2.0 * Constants.H_CGS * Constants.C_CGS*2) / (wavelength*5 * (np.exp((Constants.H_CGS * Constants.C_CGS) / (wavelength * Constants.K_CGS * temperature)) - 1))

    @staticmethod
    def tess_response_function(wavelength):
        """
        assumed uniform TESS response.
        """
        return 1  # uniform response for simplicity

    @staticmethod
    def calculate_luminosity(radius_cm, temperature):
        """
        calculation of the star's luminosity.
        """
        # star's area
        area_star = np.pi * radius_cm**2

        # integral of the Planck function
        integral, _ = quad(lambda wavelength: PhysicalCalculations.planck_function(wavelength, temperature), Constants.LAMBDA_MIN, Constants.LAMBDA_MAX)

        # calculation of the luminosity
        luminosity = area_star * integral
        return luminosity

# flares' analysis
class FlareAnalysis:
    @staticmethod
    def calculate_flare_amplitude(time_series, flare_indices, gp_curve_interp):
        """
        flare's amplitude calculation
        """
        amplitudes = []
        for index in flare_indices:
            if index >= len(gp_curve_interp):
                continue

            # getting the flux value at the time corresponding to the flare peak
            peak_flux = time_series.loc[index, 'Norm.Flux']

            #  amplitude calculation (as the difference between the peak value and the cleaned curve)
            flare_amplitude = peak_flux - gp_curve_interp[index]

            # checking for negative value
            if flare_amplitude < 0:
                flare_amplitude = 0
            amplitudes.append(flare_amplitude)
        return amplitudes

    @staticmethod
    def calculate_flare_duration(flare_indices, time_series):
        """
        calculation of the flare's duration
        """
        flare_durations = []
        for flare_index in flare_indices:
            if flare_index < len(time_series):
                background_level = np.mean(time_series['Norm.Flux'].iloc[:flare_index])
                half_max_flux = (time_series['Norm.Flux'].iloc[flare_index] + background_level) / 2
                half_max_indices = np.where(time_series['Norm.Flux'] >= half_max_flux)[0]
                fwhm_start_index = half_max_indices[half_max_indices < flare_index]
                fwhm_end_index = half_max_indices[half_max_indices > flare_index]
                if fwhm_start_index.size == 0 or fwhm_end_index.size == 0:
                    flare_durations.append(np.nan)
                else:
                    # take the first index before the peak and the last after
                    fwhm_start_index = fwhm_start_index[0]
                    fwhm_end_index = fwhm_end_index[-1]
                    start_time = time_series['TIME'].iloc[fwhm_start_index]
                    end_time = time_series['TIME'].iloc[fwhm_end_index]
                    duration_seconds = end_time - start_time
                    duration_days = duration_seconds / (24 * 3600)
                    if duration_days < 0:
                        duration_days = 0
                    flare_durations.append(duration_days)
            else:
                flare_durations.append(np.nan)
        return flare_durations

    @staticmethod
    def identify_flares(time_series, gp_curve, threshold_multiplier_range=(1.5, 3.0), num_steps=50):
        """
        identifies flares with automatic threshold selection
        """
        X = time_series['TIME'].values
        Y = time_series['Norm.Flux'].values
        gp_curve_interp = np.interp(X, np.linspace(X.min(), X.max(), len(gp_curve)), gp_curve)
        residuals = Y - gp_curve_interp
        X_train, X_test, Y_train, Y_test = train_test_split(X, residuals, test_size=0.2, random_state=42)
        best_threshold_multiplier = None
        best_accuracy = 0
        for threshold_multiplier in np.linspace(threshold_multiplier_range[0], threshold_multiplier_range[1], num_steps):
            threshold = threshold_multiplier * np.std(Y_train)
            flare_indices_train, _ = find_peaks(Y_train, height=threshold)
            threshold = threshold_multiplier * np.std(Y_test)
            flare_indices_test, _ = find_peaks(Y_test, height=threshold)
            accuracy = accuracy_score(np.isin(X_test, X_train[flare_indices_train]), np.isin(X_test, X_test[flare_indices_test]))

            # update the best threshold and accuracy
            if accuracy > best_accuracy:
                best_threshold_multiplier = threshold_multiplier
                best_accuracy = accuracy
        threshold = best_threshold_multiplier * np.std(residuals)
        flare_indices, _ = find_peaks(residuals, height=threshold)

        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        plt.show()

        return flare_indices, best_threshold_multiplier, residuals  # Возвращаем residuals

    @staticmethod
    def calculate_flare_energy(time_series, flare_indices, teff, radius, gp_curve_interp):
        """
        calculate flare energy
        """
        if time_series.empty or not np.array(flare_indices).any() or any(param is None for param in [teff, radius]):
            return np.nan

        radius_cm = radius * 6.957e10

        x = 5900. + np.arange(4100) * 1.
        y1 = PhysicalCalculations.planck_function(x, teff)
        y2 = PhysicalCalculations.planck_function(x, 9500.)
        ss = quad(lambda wavelength: PhysicalCalculations.planck_function(wavelength, teff), x[0], x[-1])[0]
        sf = quad(lambda wavelength: PhysicalCalculations.planck_function(wavelength, 9500.), x[0], x[-1])[0]
        b10 = np.pi * radius_cm**2 * ss / sf
        b2 = Constants.SIG * 9000**4 * b10

        flare_energies = []
        for index in flare_indices:
            if index not in time_series.index:
                continue
            flare_amplitude = FlareAnalysis.calculate_flare_amplitude(time_series, [index], gp_curve_interp)[0]
            duration = FlareAnalysis.calculate_flare_duration([index], time_series)[0]
            e_flare = b2 * duration * flare_amplitude
            if e_flare > 0:
                flare_energies.append(e_flare)
            else:
                flare_energies.append(np.nan)
        return flare_energies

class DataLoading:
    @staticmethod
    def create_dataframe(file_path):
        """
        create DataFrame from fits.
        """
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            df = pd.DataFrame(data, columns=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX', 'QUALITY'])
            tic_id = int(file_path.split('-')[2])
            df['TIC'] = tic_id
            return df

    @staticmethod
    def load_and_normalize_data(file_paths):
        """
        loads and normalizes data from fits files.
        """
        all_data_frames = []
        for file_path in file_paths:
            df = DataLoading.create_dataframe(file_path)
            all_data_frames.append(df)

        combined_df = pd.concat(all_data_frames)
        combined_df['Norm.Flux'] = (combined_df['SAP_FLUX'] - combined_df['SAP_FLUX'].mean()) / combined_df['SAP_FLUX'].std()
        return combined_df

    @staticmethod
    def clean_data(combined_df):
        """
        cleanes data from NaN.
        """
        selected_columns = combined_df[['TIC', 'TIME', 'Norm.Flux', 'PDCSAP_FLUX']]
        selected_columns_cleaned = selected_columns.dropna()
        return selected_columns_cleaned

class ResultsSaving:
    @staticmethod
    def save_flare_data(flare_data, save_path):
        """
        saves flare data
        """
        output_file_path = os.path.join(save_path, 'flares_data.txt')
        with open(output_file_path, 'w') as output_file:
            output_file.write("FlareNumber TIC Time Amplitude Duration Energy\n")
            for flare_number, (tic, time, amplitude, duration, energy) in enumerate(flare_data, 1):
                output_file.write(f"{flare_number} "
                                  f"{tic:9d} "
                                  f"{time:9.6f} "
                                  f"{amplitude:8.5f} "
                                  f"{duration:8.5f} "
                                  f"{energy:9.2e}\n")

class DataProcessor:
    def _init_(self, file_paths, save_path):
        self.file_paths = file_paths
        self.save_path = save_path
        self.gp_model = GaussianProcessModel()
        self.flare_analysis = FlareAnalysis()

    def process_data(self):
        """
        flare data processing
        """
        combined_df = DataLoading.load_and_normalize_data(self.file_paths)
        selected_columns_cleaned = DataLoading.clean_data(combined_df)

        output_file_path = os.path.join(self.save_path, 'flares_data.txt')
        with open(output_file_path, 'w') as output_file:
            output_file.write("FlareNumber TIC Time Amplitude Duration Energy\n") #  Добавлен столбец TIC в заголовок
            num_sets = 10
            set_size = len(combined_df) // num_sets
            flare_number_counter = 0
            fig, ax = plt.subplots(figsize=(10, 6))

            for i in range(num_sets):
                set_start = i * set_size
                set_end = (i + 1) * set_size
                set_data = selected_columns_cleaned.iloc[set_start:set_end].copy()
                set_data.reset_index(drop=True, inplace=True)
                if set_data[['TIME', 'Norm.Flux']].isnull().values.any():
                    set_data = set_data.dropna(subset=['TIME', 'Norm.Flux'])
                if set_data.empty:
                    continue

                set_data.loc[:, 'Norm.Flux'] = Filters.butter_bandpass_filter(set_data['Norm.Flux'],
                                                                  lowcut=0.1, highcut=10.0, fs=120, order=5) # example

                gp_curve = self.gp_model.fit_and_predict(set_data, plot=True)
                flare_indices, best_threshold_multiplier, residuals = self.flare_analysis.identify_flares(set_data, gp_curve)

                if flare_indices.size > 0:
                    gp_curve_interp = np.interp(set_data['TIME'].values, np.linspace(set_data['TIME'].min(), set_data['TIME'].max(), len(gp_curve)), gp_curve)

                    flare_duration = self.flare_analysis.calculate_flare_duration(flare_indices, set_data)
                    flare_energies = self.flare_analysis.calculate_flare_energy(set_data, flare_indices, teff, radius, gp_curve_interp)
                    flare_amplitudes = self.flare_analysis.calculate_flare_amplitude(set_data, flare_indices, gp_curve_interp)

                    for j, flare_index in enumerate(flare_indices):
                        flare_number_counter += 1
                        if flare_index < len(set_data):
                            if flare_amplitudes:
                                flare_amplitude = flare_amplitudes[min(j, len(flare_amplitudes)-1)]
                            else:
                                flare_amplitude = np.nan
                            flare_energy = flare_energies[j] if j < len(flare_energies) else np.nan

                            if flare_amplitude > 0 and flare_energy > 0 and not np.isnan(flare_amplitude) and not np.isnan(flare_energy):
                                tic = int(set_data.iloc[flare_index]['TIC'])
                                output_file.write(f"{flare_number_counter} "
                                                  f"{tic:9d} "
                                                  f"{set_data.loc[flare_index, 'TIME']:9.6f} "
                                                  f"{flare_amplitude:8.5f} "
                                                  f"{flare_duration[j]:8.5f} "
                                                  f"{flare_energy:9.2e}\n")

                            ax.scatter(set_data['TIME'], set_data['Norm.Flux'], c='b', s=2, alpha=0.5)
                            ax.scatter(set_data['TIME'].iloc[flare_indices], set_data['Norm.Flux'].iloc[flare_indices], c='r', s=20, label='Flare')

                            for j, flare_index in enumerate(flare_indices):
                                if j < len(flare_amplitudes) and flare_amplitudes[j] > 0:
                                    start_time = set_data['TIME'].iloc[flare_index] - 0.01
                                    end_time = set_data['TIME'].iloc[flare_index] + 0.01
                                    gp_curve_values = gp_curve_interp[(set_data['TIME'] >= start_time) & (set_data['TIME'] <= end_time)]
                                    ax.fill_between(set_data['TIME'][
                                                        (set_data['TIME'] >= start_time) & (set_data['TIME'] <= end_time)],
                                                        gp_curve_values,
                                                        set_data['Norm.Flux'][
                                                            (set_data['TIME'] >= start_time) & (set_data['TIME'] <= end_time)],
                                                        color='pink', alpha=0.5)
                            print(f"threshold value for set {i + 1}: {best_threshold_multiplier:.2f}")
                            plt.figure(figsize=(10, 6))
                            plt.hist(residuals, bins=50)
                            plt.xlabel('Residuals')
                            plt.ylabel('Frequency')
                            plt.title(f'Histogram of Residuals (Set {i + 1})')
                            plt.show()
                    ax.scatter(set_data['TIME'], set_data['Norm.Flux'], c='b', s=2, alpha=0.5)
                    ax.scatter(set_data['TIME'].iloc[flare_indices], set_data['Norm.Flux'].iloc[flare_indices], c='r', s=20, label='Flare')
                    for j, flare_index in enumerate(flare_indices):
                        if j < len(flare_amplitudes) and flare_amplitudes[j] > 0:
                            start_time = set_data['TIME'].iloc[flare_index] - 0.01
                            end_time = set_data['TIME'].iloc[flare_index] + 0.01
                            gp_curve_values = gp_curve_interp[(set_data['TIME'] >= start_time) & (set_data['TIME'] <= end_time)]
                            ax.fill_between(set_data['TIME'][
                                                (set_data['TIME'] >= start_time) & (set_data['TIME'] <= end_time)],
                                                gp_curve_values,
                                                set_data['Norm.Flux'][
                                                    (set_data['TIME'] >= start_time) & (set_data['TIME'] <= end_time)],
                                                color='pink', alpha=0.5)
                    print(f"threshold value for set {i + 1}: {best_threshold_multiplier:.2f}")
                    plt.figure(figsize=(10, 6))
                    plt.hist(residuals, bins=50)
                    plt.xlabel('Residuals')
                    plt.ylabel('Frequency')
                    plt.title(f'Histogram of Residuals (Set {i + 1})')
                    plt.show()
            ax.set_xlabel('Time')
            ax.set_ylabel('Norm.Flux')
            ax.set_title('Original Data with Identified Flares')
            ax.legend()
            plt.show()

        output_file.close()

#  example of use
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
