import numpy as np
from loguru import logger
from sortedcontainers import SortedList
from mlflow.pyfunc import PythonModel

class PeriodCollaborativeFiltering(PythonModel):
    """
    Class for computing period similarities using collaborative filtering.
    """

    def __init__(self, period2farm, periodfarm2power, lst_periods, K, min_common_farms):
        """
        Initialize the PeriodCollaborativeFiltering instance.

        Parameters:
        - period2farm (dict): A dictionary mapping periods to corresponding farms.
        - periodfarm2power (dict): A dictionary mapping (period, farm) tuples to power values.
        - lst_periods (list): A list of periods.
        - K (int): Number of closest periods to consider.
        - min_common_farms (int): Minimum number of common farms for similarity consideration.
        """
        self.period2farm = period2farm
        self.periodfarm2power = periodfarm2power
        self.lst_periods = lst_periods
        self.K = K
        self.min_common_farms = min_common_farms
        self.neighbors = {}
        self.averages = {}
        self.deviations = {}
        self.sigmas = {}

    @staticmethod
    def compute_rmse(predictions, targets):
        predictions = np.array(predictions)
        targets = np.array(targets)
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        return rmse

    def calculate_avg_and_deviation(self, period, farms):
        """
        Calculate average power and deviation for a given period and its corresponding farms.

        Parameters:
        - period (object): The period for which average and deviation are calculated.
        - farms (list): List of farms corresponding to the period.

        Returns:
        - avg_power (float): Average power for the period.
        - dev_power (dict): Deviation of power for each farm.
        - sigma_power (float): Standard deviation of power for the period.
        """
        try:
            powers = {farm: self.periodfarm2power[(period, farm)] for farm in farms}
            avg_power = np.mean(list(powers.values()))
            dev_power = {farm: (power - avg_power) for farm, power in powers.items()}
            dev_power_values = np.array(list(dev_power.values()))
            sigma_power = np.sqrt(dev_power_values.dot(dev_power_values))
            return avg_power, dev_power, sigma_power
        except Exception as e:
            print(f"Error in calculating average and deviation for period {period}: {e}")
            return None

    def compute_period_similarities(self):
        """
        Compute period similarities based on collaborative filtering.

        Returns:
        - neighbors (dict): A dictionary mapping periods to their K closest neighbors.
        - averages (dict): A dictionary mapping periods to their average power.
        - deviations (dict): A dictionary mapping periods to their power deviations.
        - sigmas (dict): A dictionary mapping periods to their power standard deviations.
        """
        try:
            count=0
            for period_i in self.lst_periods:
                farms_i = self.period2farm[period_i]
                farms_i_set = set(farms_i)
                avg_power_i, dev_power_i, sigma_power_i = self.calculate_avg_and_deviation(period_i, farms_i)
                self.averages[period_i] = avg_power_i
                self.deviations[period_i] = dev_power_i
                self.sigmas[period_i] = sigma_power_i
                sl = SortedList()
                for period_j in self.lst_periods:
                    if period_j != period_i:
                        farms_j = self.period2farm[period_j]
                        farms_j_set = set(farms_j)
                        common_farms = farms_i_set & farms_j_set
                        if len(common_farms) >= self.min_common_farms:
                            _, dev_power_j, sigma_power_j = self.calculate_avg_and_deviation(period_j, farms_j)
                            numerator = sum(dev_power_i[farm] * dev_power_j[farm] for farm in common_farms)
                            w_ij = numerator / (sigma_power_i * sigma_power_j)
                            sl.add((-w_ij, period_j))
                            if len(sl) > self.K:
                                del sl[-1]
                self.neighbors[period_i] = sl
                if count % 5000 == 0:
                    logger.info(f'Total processed periods: {count}')
                count+=1
            logger.info(f'Total processed periods: {count}')
            return self.neighbors, self.averages, self.deviations, self.sigmas
        except Exception as e:
            print(f"Error in computing period similarities: {e}")
            return None

    def compute_predictions(self, period_i, farm_m):
            """
            Compute power prediction for a given period and farm using collaborative filtering.

            Parameters:
            - period_i (object): The period for which the prediction is computed.
            - farm_m (object): The farm for which the prediction is computed.

            Returns:
            - prediction (float): The predicted power value.
            """
            try:
                numerator = 0
                denominator = 0
                for neg_w, period_j in self.neighbors[period_i]:
                    try:
                        numerator += -neg_w * self.deviations[period_j][farm_m]
                        denominator += abs(neg_w)
                    except KeyError:
                        pass
                if denominator == 0:
                    prediction = self.averages[period_i]
                else:
                    prediction = self.averages[period_i] + numerator / denominator
                prediction = min(1, prediction)  # max power is 1.0 in normalized data
                prediction = max(0, prediction)  # min power is 0.0
                return prediction
            except Exception as e:
                print(f"Error in computing prediction for period {period_i} and farm {farm_m}: {e}")
                return None

    def predict(self, power_dict):
            """
            Make power predictions for a dictionary of period-farm pairs.

            Parameters:
            - power_dict (dict): A dictionary mapping (period, farm) tuples to target power values.

            Returns:
            - predictions (list): List of predicted power values.
            - targets (list): List of target power values.
            - periods (list): List of periods corresponding to the predictions.
            - farms (list): List of farms corresponding to the predictions.
            """
            try:
                predictions = []
                targets = []
                periods = []
                farms = []
                for (period_i, farm_m), target_power in power_dict.items():
                    prediction = self.compute_predictions(period_i, farm_m)
                    predictions.append(prediction)
                    targets.append(target_power)
                    periods.append(period_i)
                    farms.append(farm_m)
                return predictions, targets, periods, farms
            except Exception as e:
                print(f"Error in making predictions: {e}")
                return None
