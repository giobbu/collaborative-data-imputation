from mlflow.pyfunc import PythonModel
import numpy as np
from sortedcontainers import SortedList
from loguru import logger

class FarmCollaborativeFiltering(PythonModel):
    def __init__(self, farm2period, periodfarm2power, lst_farms, K, min_common_periods):
        self.farm2period = farm2period
        self.periodfarm2power = periodfarm2power
        self.lst_farms = lst_farms
        self.K = K
        self.min_common_periods = min_common_periods
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

    def calculate_avg_and_deviation(self, farm, periods):
        try:
            powers = {period: self.periodfarm2power[(period, farm)] for period in periods}
            avg_power = np.mean(list(powers.values()))
            dev_power = {period: (power - avg_power) for period, power in powers.items()}
            dev_power_values = np.array(list(dev_power.values()))
            sigma_power = np.sqrt(dev_power_values.dot(dev_power_values))
            return avg_power, dev_power, sigma_power
        except Exception as e:
            logger.error(f"Error in calculating average and deviation for farm {farm}: {e}")
            return None

    def compute_similarities(self):
        try:
            count=0
            logger.info('Start Data Imputation based on Wind Farms Similarity') 
            logger.info('List of Wind Farms ' + str(self.lst_farms))
            for farm_i in self.lst_farms:
                periods_i = self.farm2period[farm_i]
                periods_i_set = set(periods_i)
                avg_power_i, dev_power_i, sigma_power_i = self.calculate_avg_and_deviation(farm_i, periods_i)
                self.averages[farm_i] = avg_power_i
                self.deviations[farm_i] = dev_power_i
                self.sigmas[farm_i] = sigma_power_i
                sl = SortedList()
                for farm_j in self.lst_farms:
                    if farm_j != farm_i:
                        periods_j = self.farm2period[farm_j]
                        periods_j_set = set(periods_j)
                        common_periods = periods_i_set & periods_j_set
                        if len(common_periods) >= self.min_common_periods:
                            _, dev_power_j, sigma_power_j = self.calculate_avg_and_deviation(farm_j, periods_j)
                            numerator = sum(dev_power_i[period] * dev_power_j[period] for period in common_periods)
                            w_ij = numerator / (sigma_power_i * sigma_power_j)
                            sl.add((-w_ij, farm_j))
                            if len(sl) > self.K:
                                del sl[-1]
                self.neighbors[farm_i] = sl
                if count % 5 == 0:
                    logger.info('Wind Farms Processed: ' + str(count))
                count+=1
            return self.neighbors, self.averages, self.deviations, self.sigmas
        except Exception as e:
            logger.error(f"Error in computing period similarities: {e}")
            return None

    def compute_predictions(self, farm_i, period_m):
            try:
                numerator = 0
                denominator = 0
                for neg_w, farm_j in self.neighbors[farm_i]:
                    try:
                        numerator += -neg_w * self.deviations[farm_j][period_m]
                        denominator += abs(neg_w)
                    except KeyError:
                        pass
                if denominator == 0:
                    prediction = self.averages[farm_i]
                else:
                    prediction = self.averages[farm_i] + numerator / denominator
                prediction = min(1, prediction)  # max power is 1.0 in normalized data
                prediction = max(0, prediction)  # min power is 0.0
                return prediction
            except Exception as e:
                logger.error(f"Error in computing prediction for farm {farm_i} and farm {period_m}: {e}")
                return None

    def predict(self, power_dict):
        try:
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
        except Exception as e:
            logger.error(f"Error in making predictions: {e}")
            return None

