
import process_input
import Population_Process



# ------------------------------------------------------------------------------------------------
class main():

    descriptors_file = "Practice_Descriptors.csv"
    targets_file = "Practice_Targets.csv"


    def __init__(self):
    #class constructor
    #object inputt from class process used to load data and clean, sort, rescale the data
        self.inputt = process_input.process()
        while True:
            try:
                self.number = int(input('please enter 0 to do LMR,1 to do SVM and 2 to do ANN on this dataset: '))
                assert  (self.number== 0) or (self.number == 1) or (self.number == 2)
            except ValueError:

                print("Not an integer! Please enter an integer.")
            except AssertionError:
                    print("Please enter an integer among 0 , 1 or 2")
            else:
                break
    # ------------------------------------------------------------------------------------------------
    # Step 1
    def step1(self):
    #load descriptors and target from file
        self.descriptors = self.inputt.open_descriptor_matrix(self.descriptors_file)
        self.targets = self.inputt.open_target_values(self.targets_file)
    # ------------------------------------------------------------------------------------------------
    # Step 2
    # Filter out molecules with NaN-value descriptors and descriptors with little or no variance
    def step2(self):
        self.descriptors, self.targets = self.inputt.removeInvalidData(self.descriptors, self.targets)
        self.descriptors, self.active_descriptors = self.inputt.removeNearConstantColumns(self.descriptors)
        # Rescale the descriptor data
        self.descriptors = self.inputt.rescale_data(self.descriptors)

    # ------------------------------------------------------------------------------------------------
    # Step 3
    def step3(self):
        self.descriptors, self.targets = self.inputt.sort_descriptor_matrix(self.descriptors, self.targets)

    # ------------------------------------------------------------------------------------------------
    # Step 4
    def step4(self):
        #splite data  to test , validation,train to use in  evaluation function and calculate fitness and errors
        self.X_Train, self.X_Valid, self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test = self.inputt.simple_split(self.descriptors, self.targets)
        self.data = {'TrainX': self.X_Train, 'TrainY': self.Y_Train, 'ValidateX': self.X_Valid, 'ValidateY': self.Y_Valid,
              'TestX': self.X_Test, 'TestY': self.Y_Test, 'UsedDesc': self.active_descriptors}

        print(str(self.descriptors.shape[1]) + " valid descriptors and " + str(self.targets.__len__()) + " molecules available.")



    # ------------------------------------------------------------------------------------------------
    # Step 5
    def step5(self):
        #create an object from PBSO_Algorithm class to use all PBSO functions in calculating best and optimum fitness and errors
        self.PBSO = Population_Process.PBSO_Algorithm(self.number, self.data)
        # create initial population

    def step6(self):
        velocity = self.PBSO.createInitialVelocity()
        population = self.PBSO.initial_Population_total()


        alpha = 0.5
        for j in range(10000):
            # Create a Multiple Linear Regression object to fit our demonstration model to the data in modeling function
            df= self.PBSO.modeling(population , j)
            # save result to file and print best fitness with the number of generation on the screen

            if j == 0:
                localBestMatrix, localBestFitness= self.PBSO.initialLocalBestMatrix(population, df)
                globalBestRow = self.PBSO.updateGlobalBestRow(localBestMatrix, localBestFitness,j)

            else:
                localBestMatrix, localBestFitness = self.PBSO.updateLocalBestMatrix(localBestMatrix, localBestFitness, population , df)
                globalBestRow= self.PBSO.updateGlobalBestRow(localBestMatrix, localBestFitness,j)
                velocity = self.PBSO.updateVelocity(velocity, localBestMatrix, globalBestRow, population)

            alpha = alpha - (0.5 - 0.33) / (10000 + 1)
            #create next population based on new a value and p that are calulated based on j in each iteration

            population = self.PBSO.createNextPopulation(alpha, localBestMatrix,velocity, globalBestRow, population)
        self.PBSO.removeRedunduntFromFile()

        # ------------------------------------------------------------------------------------------------

#call constructor of main class to creat an object (m) and then call each step's routins back to back
m = main()
m.step1()
m.step2()
m.step3()
m.step4()
m.step5()
m.step6()
