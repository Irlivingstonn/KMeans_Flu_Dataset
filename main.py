# Name: Isabella Livingston

# Importing Assets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# EuclideanDistance Function: Calculates the distance of two different points from eachother
# Borrowed code from: https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Initialize Centroids Function: Randomly initializes the positions of the k cluster centroids
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)

    # Returns a list of the centroids
    return X[indices]

# Finding Closest Centroid to Student Function: Tries to find which centroid is closest to the Student
def finding_closest_centroid_to_student(student, initialized_centroids, k):

    # Initalizing variables
    # I decided to set a high value to set a high value for the min value, that way the actual smallest value can replace it
    min_value = 100000
    closest_centroid_index = -1

    # Checks all k values
    for element in range(k):
        # Calculates the distance of a student to the centroid
        dist_to_centroid = euclidean_distance(student, initialized_centroids[element])

        # If the Distance to the centroid value is smaller than the min value
        # Then the min value is set to the centroid value
        if dist_to_centroid < min_value:
            min_value = dist_to_centroid
            closest_centroid_index = element

    # Returns a list of centroids that are closest to the students based on index
    return closest_centroid_index

# Assign Clusters Function: Assigns the clusters to the students
def assign_clusters(list_of_students, initialized_centroids, k):

    # Creates an empty list to store the centroids, and finds the closest centroid to the student
    centroid_list = []
    for student in list_of_students:
        closest_centroid_index = finding_closest_centroid_to_student(student, initialized_centroids, k)

        # I figured since the data isn't being manipulated for the index for the
        # total amount of students would change. I thought it would be okay to put
        # all centroid indexes in a list and search it by its list's index
        centroid_list.append(closest_centroid_index)

    # Returns the list of centroids closest to the students
    return centroid_list

# Update Centroids Function
def update_centroids(list_of_students, initialized_centroids, k, centroid_list):

    # Creates an empty list to store the new centroids
    new_centroids = []

    # Loops based on the number of k
    for k_num in range(k):

        # Finds all students assigned to cluster k_num
        cluster_points = []

        # i is the index in the centroid list. And cluster assignment is the value in the array
        for i, cluster_assignment in enumerate(centroid_list):

            # Checks if the cluster assignment is the same value as the k value
            if cluster_assignment == k_num:

                # Appends the element to the cluster points
                cluster_points.append(list_of_students[i])

        # Tries to compute mean of cluster_points
        if len(cluster_points) > 0:
            cluster_points = np.array(cluster_points)
            mean_point = np.mean(cluster_points, axis=0)

            new_centroids.append(mean_point)

        # If there are no points assigned to this cluster, then we keep the old centroid
        else:
            new_centroids.append(initialized_centroids[k_num])

    # Sets the new centroids to the current centroids
    initialized_centroids = np.array(new_centroids)

    # Returns the initialized centroids
    return initialized_centroids


# Plot clusters function: 3d Scatter Plot
def plot_clusters(list_of_students, centroid_list, updated_centroids, k):

    # Creates a figure and the 3d axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Tries to set the features to different axis of the 3d graph
    x = list_of_students[:, 0] # Risk
    y = list_of_students[:, 1] # NoFaceContact
    z = list_of_students[:, 2] # Sick

    # Plots the points - specifically by what cluster they're assigned to
    scatter = ax.scatter3D(x, y, z, c=centroid_list, cmap='viridis', marker='o')

    # Tries to plot the centroids
    centroids_x = updated_centroids[:, 0] # Risk
    centroids_y = updated_centroids[:, 1] # NoFaceContact
    centroids_z = updated_centroids[:, 2] # Sick

    ax.scatter3D(centroids_x, centroids_y, centroids_z, color='red', marker='X', s=200, label='Centroids')

    # Labels the axises
    ax.set_xlabel('Risk')
    ax.set_ylabel('NoFaceContact')
    ax.set_zlabel('Sick')
    ax.set_title('3D Scatter Plot of K-Means Clusters (k = ' + str(k) + ')')

    # Shows the graph
    plt.show()

# Getting Max intra dist Function: gets the max intra distance
def getting_max_intra_dist(centroid_list, list_of_students, k):
    max_intra_dist = 0

    # Goes through the values of k
    for k_num in range(k):

        # Get points in the current cluster
        cluster_points = []

        # Same code to when we tried to update the centroids
        for i, cluster_assignment in enumerate(centroid_list):
            if cluster_assignment == k_num:
                cluster_points.append(list_of_students[i])

        # Tries to compare all pairs inside the current cluster
        for i in range(len(cluster_points)):

            # Set the range from i to the length of the cluster_points
            # to avoid redundant code for calculating the distance
            # Ex.) euclidean_distance(a, b) and euclidean_distance(b, a)
            for j in range(i + 1, len(cluster_points)):

                dist = euclidean_distance(cluster_points[i], cluster_points[j])
                if dist > max_intra_dist:
                    max_intra_dist = dist

    # Returns the max intra distance
    return max_intra_dist

# Getting min inter distance function: Gets the minimum inter distance value
def getting_min_inter_dist(updated_centroids, k):

    # Initializes the variable
    min_inter_dist = float('inf')

    # Looks through all the k values
    for i in range(k):

        # Same code as to when we tried to get the max intra dist value (but for min inter)
        for j in range(i + 1, k):
            dist = euclidean_distance(updated_centroids[i], updated_centroids[j])

            # Gets the smallest distance and sets it to the min inter distance value
            if dist < min_inter_dist:
                min_inter_dist = dist

    # Returns the min inter distance
    return min_inter_dist


# Calculating dunn index value: Tries to calculate the Dunn Index value
def calculating_dunn_index(centroid_list, list_of_students, updated_centroids, k):
    # Gets the max intra distance and min inter distance value
    max_intra_dist = getting_max_intra_dist(centroid_list, list_of_students, k)
    min_inter_dist = getting_min_inter_dist(updated_centroids, k)

    # Calculates and returns the dunn intex value
    dunn_index = min_inter_dist / max_intra_dist

    return dunn_index

# K means function: This function runs the k means algorithm (followed this algorithm to the one shown in class)
def k_means(list_of_students, repeat, k):

    # Randomly selects 2 different data points to be used as the centroids
    initialized_centroids = initialize_centroids(list_of_students, k)

    # Loops until it finishes converging
    for x in range(repeat):

        # Assigns the clusters
        centroid_list = assign_clusters(list_of_students, initialized_centroids, k)

        # Updates the Centroids
        updated_centroids = update_centroids(list_of_students, initialized_centroids, k, centroid_list)

    # Calculates the Dunn Index
    dunn_index = calculating_dunn_index(centroid_list, list_of_students, updated_centroids, k)

    # Displays it to the user
    print("(k-means) For k=" + str(k) + ", Dunn Index=" + str(dunn_index))

    # Plots the cluster data
    plot_clusters(list_of_students, centroid_list, updated_centroids, k)


# Initalize U matrix: Creates the U matrix
def initialize_u_matrix(list_of_students, k):
    # Converts the data frame into an array
    list_of_students = np.array(list_of_students)
    u_matrix = np.random.rand(list_of_students.shape[0], k)

    # Normalizing the data to be between 0 and 1, so that student info can add up to 1
    u_matrix = u_matrix / np.sum(u_matrix, axis=1, keepdims=True)

    # Returns the u matrix
    return u_matrix


# Updates centroids fuzzy function: Updates the centroids but for the fuzzy c-means algorithm
def update_centroids_fuzzy(list_of_students, u_matrix, m):

    # Gets the number of features
    num_features = list_of_students.shape[1]

    # Creating an empty array to hold the centroids
    centroids = np.zeros((u_matrix.shape[1], num_features))

    # Loops through each cluster (j = cluster index)
    for j in range(u_matrix.shape[1]):

        # Calucates the numerator: sum of (membership^m * data point), across all of the data points
        numerator = np.sum((u_matrix[:, j][:, np.newaxis] ** m) * list_of_students, axis=0)

        # Calculates the denomiator: sum of membership^m for this cluster
        denominator = np.sum(u_matrix[:, j] ** m)

        # Calculates the new centroid as weighted average
        centroids[j] = numerator / denominator

    # Returns the new centroids
    return centroids

# Update u_matrix function: Updates the membership matrix (u matrix)
def update_u_matrix(list_of_students, current_centroids, m):

    # Initialzing the variables
    num_points = list_of_students.shape[0] # Number of data points
    k = current_centroids.shape[0] # Number of clusters
    new_u_matrix = np.zeros((num_points, k)) # Initialize new u matrix

    # Loops through each data point
    for i in range(num_points):

        # Loops through each cluster
        for j in range(k):

            # Initializes the denominator sum
            denom_sum = 0.0

            # Gets the distance from point i to centroid j
            dist_i_to_j = euclidean_distance(list_of_students[i], current_centroids[j]) + 1e-8  # avoid division by zero

            # Loops through all clusters again to compute denominator of the formula
            for l in range(k):
                dist_i_to_l = euclidean_distance(list_of_students[i], current_centroids[l]) + 1e-8
                denom_sum += (dist_i_to_j / dist_i_to_l) ** (2 / (m - 1))

            # Updates the value for point i to cluster j
            new_u_matrix[i, j] = 1.0 / denom_sum

    # Returns the new u matrix
    return new_u_matrix

# Fuzzy c means function: Runs the Fuzzy c means Algorithm
def fuzzy_c_means(list_of_students, repeat, k, m):

    # Initializes the u matrix
    u_matrix = initialize_u_matrix(list_of_students, k)

    # Repeats until convergence
    for x in range(repeat):
        # Updates the fuzzy centroids
        current_centroids = update_centroids_fuzzy(list_of_students, u_matrix, m)

        # Updates the U matrix
        u_matrix = update_u_matrix(list_of_students, current_centroids, m)

    # Tries to harden for visualization to assign each point to cluster with the highest value
    centroid_list_hardened = np.argmax(u_matrix, axis=1)

    # Calculates the Dunn Index
    dunn_index = calculating_dunn_index(centroid_list_hardened, list_of_students, current_centroids, k)

    print("(Fuzzy c-means) For k=" + str(k) + ", Dunn Index=" + str(dunn_index) + ")")

    # Plot hardened clusters
    plot_clusters(list_of_students, centroid_list_hardened, current_centroids, k)

# Main Program
def main():

    # Gets the data from the csv file
    data = pd.read_csv('flu_data.csv', sep= ',')

    # I noticed that I was missing 36 values for Sick out of 410 students.
    # Since this makes up 8.78% of the total dataset, I've decided to not use that data.
    cleaned_data = data.dropna(subset=['Risk', 'NoFaceContact', 'Sick']) # , 'KnowlTrans', 'HndWshQual'

    # I also noticed that for NoFaceContact there was 1 outlier that's value was equal to 9
    # I put this here to remove the outlier so the clustering is more accurate
    cleaned_data = cleaned_data[cleaned_data['NoFaceContact'] != 9]

    # Puts the data into a list, based on Risk, NoFaceContact, Sick, KnowlTrans, HndWshQual values
    list_of_students = cleaned_data[['Risk', 'NoFaceContact', 'Sick']].values # , 'KnowlTrans', 'HndWshQual'

    # Initialzing variables
    k = 2
    m = 2

    repeat = 800

    for k_num in range(2, 11):
        k = k_num
        fuzzy_c_means(list_of_students, repeat, k, m)

    #for k_num in range(2, 11):
    #    k = k_num
    #    k_means(list_of_students, repeat, k)

# Runs Main Program
main()
