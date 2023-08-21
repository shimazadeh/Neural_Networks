    # #create a new table with the statistics
    # table = []
    # table.append(["", "count", "mean", "std", "skew", "kurtosis", "variance", "min", "25%", "50%", "75%", "max"])
    # min_std_cat = "not found"
    # min_std = np.inf

    # for idx, (column_name, column_data) in enumerate(data.items()):            
    #     if np.issubdtype(column_data.dtype, np.number) and idx != 0:
    #         name = column_name[:5]
    #         count = column_data.count()
    #         mean = round(column_data.mean(), 4)
    #         std = round(column_data.std(), 4)
    #         skewness = round(skew(column_data), 4)
    #         kurt = round(kurtosis(column_data), 4)
    #         variance = round(column_data.var(), 4)
    #         min_val = round(column_data.min(), 4)
    #         percentile_25 = round(np.percentile(column_data, 25), 4)
    #         percentile_50 = round(np.percentile(column_data, 50), 4)
    #         percentile_75 = round(np.percentile(column_data, 75), 4)
    #         max_val = round(column_data.max(), 4)

    #         # Append the column to the new table
    #         table.append([name, count, mean, std, variance, skewness, kurt, min_val, percentile_25, percentile_50, percentile_75, max_val])

    #         #check for minimum
    #         if (std < min_std):
    #             min_std_cat = column_name
    #             min_std = std

    # #print the table
    # arr = np.array(table, dtype=object) 
    # save_output(np.transpose(arr), "stats_output.txt")

    # # section 2: calculating the correlation between every two feature
    # data_wo = data.drop(columns=["id"])
    # Feature_names = data_wo.columns

    # correlation_matrix = [[0] * (len(Feature_names) + 1) for _ in range(len(Feature_names) + 1)]
    # correlation_matrix[0] = (list(Feature_names))

    # # Initialize the remaining cells with zeros
    # for i in range(1, len(correlation_matrix)):
    #     correlation_matrix[i][0] = Feature_names[i - 1]
            
    # for i in range(1, len(correlation_matrix)):
    #     for j in range(i, len(correlation_matrix[0])):
    #         feature1 = correlation_matrix[i][0]
    #         feature2 = correlation_matrix[0][j]
    #         # print(i, j, feature1, feature2)
    #         correlation_matrix[i][j] = data[feature1].corr(data[feature2])
    #         correlation_matrix[j][i] = correlation_matrix[i][j]

    # save_output(correlation_matrix, "corr_output.txt")
  