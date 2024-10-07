from mlflow.pyfunc import PythonModel
import numpy as np
from sortedcontainers import SortedList
from loguru import logger

class FarmCollaborativeFiltering(PythonModel):
    " Collaborative Filtering model based on wind farms similarity. "
    
    def __init__(self, farm2period, periodfarm2power, lst_farms, K, min_common_periods):
        " Initialize the FarmCollaborativeFiltering instance. "
        # Check if the inputs are of the correct type
        assert isinstance(farm2period, dict), "Input farm2period must be a dictionary."
        assert isinstance(periodfarm2power, dict), "Input periodfarm2power must be a dictionary."
        assert isinstance(lst_farms, list), "Input lst_farms must be a list."
        assert isinstance(K, int), "Input K must be an integer."
        assert isinstance(min_common_periods, int), "Input min_common_periods must be an integer."
        # Store the inputs
        self.farm2period = farm2period
        self.periodfarm2power = periodfarm2power
        self.lst_farms = lst_farms
        self.K = K
        self.min_common_periods = min_common_periods
        self.neighbors = {}
        self.averages = {}
        self.deviations = {}
        self.sigmas = {}

    def calculate_avg_and_deviation(self, farm, periods):
        " Calculate the average, deviation, and sigma of power for a given farm. "
        powers = {period: self.periodfarm2power[(period, farm)] for period in periods}
        # Compute the average power
        avg_power = np.mean(list(powers.values()))
        # Compute the deviation of power
        dev_power = {period: (power - avg_power) for period, power in powers.items()}
        dev_power_values = np.array(list(dev_power.values()))
        # Compute the sigma of power
        sigma_power = np.sqrt(dev_power_values.dot(dev_power_values))
        return avg_power, dev_power, sigma_power

    def compute_similarities(self):
        " Compute the similarities between wind farms. "
        logger.info('Start Data Imputation based on Wind Farms Similarity') 
        logger.info('List of Wind Farms ' + str(self.lst_farms))
        # Iterate over the wind farms
        count=0
        for farm_i in self.lst_farms:
            periods_i = self.farm2period[farm_i]  # Get the periods of farm_i
            periods_i_set = set(periods_i)  # Convert periods_i to a set
            # Calculate the average, deviation, and the sigma of power for farm_i
            avg_power_i, dev_power_i, sigma_power_i = self.calculate_avg_and_deviation(farm_i, periods_i)
            self.averages[farm_i] = avg_power_i  # Store the average power of farm_i
            self.deviations[farm_i] = dev_power_i  # Store the deviation of power of farm_i
            self.sigmas[farm_i] = sigma_power_i  # Store the sigma of power of farm_i
            # Create a SortedList to store the similarities
            sl = SortedList()
            # Iterate over the other wind farms
            for farm_j in self.lst_farms:
                # Skip farm_i
                if farm_j != farm_i:
                    periods_j = self.farm2period[farm_j]  # Get the periods of farm_j
                    periods_j_set = set(periods_j)  # Convert periods_j to a set
                    common_periods = periods_i_set & periods_j_set  # Find the common periods
                    if len(common_periods) >= self.min_common_periods:  # Check if the number of common periods is greater than min_common_periods
                        _, dev_power_j, sigma_power_j = self.calculate_avg_and_deviation(farm_j, periods_j)
                        # Compute the similarity between farm_i and farm_j
                        numerator = sum(dev_power_i[period] * dev_power_j[period] for period in common_periods)
                        w_ij = numerator / (sigma_power_i * sigma_power_j)
                        # Add the similarity to the SortedList
                        sl.add((-w_ij, farm_j))
                        # Keep only the K most similar wind farms
                        if len(sl) > self.K:
                            del sl[-1]
            # Store the K most similar wind farms
            self.neighbors[farm_i] = sl
            # Log the progress
            if count % 5 == 0:
                logger.info('Wind Farms Processed: ' + str(count))
            count += 1  # Increment the count
        return self.neighbors, self.averages, self.deviations, self.sigmas

    def compute_predictions(self, farm_i, period_m):
            " Compute the prediction for a given farm and period. "
            # Iterate over the neighbors of farm_i
            # Compute the numerator and denominator for the prediction
            numerator = 0
            denominator = 0
            for neg_w, farm_j in self.neighbors[farm_i]:
                try:
                    # Check if farm_j has a power value for period_m
                    # If not, skip the farm
                    numerator += -neg_w * self.deviations[farm_j][period_m]
                    denominator += abs(neg_w)
                except KeyError:
                    pass
            # Compute the prediction
            if denominator == 0:
                # If denominator is 0, use the average power of farm_i
                prediction = self.averages[farm_i]
            else:
                # Compute the prediction
                prediction = self.averages[farm_i] + numerator / denominator
            prediction = min(1, prediction)  # max power is 1.0 in normalized data
            prediction = max(0, prediction)  # min power is 0.0
            return prediction

    def predict(self, power_dict):
        " Predict using the model. "
        predictions = []
        targets = []
        periods = []
        farms = []
        for (period_m, farm_i), target_power in power_dict.items():
            if farm_i in self.lst_farms:
                prediction = self.compute_predictions(farm_i, period_m)
                predictions.append(prediction)
                targets.append(target_power)
                periods.append(period_m)
                farms.append(farm_i)
        return predictions, targets, periods, farms

    @staticmethod
    def compute_rmse(predictions, targets):
        " Compute the root mean squared error. "
        predictions = np.array(predictions)
        targets = np.array(targets)
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        return rmse

