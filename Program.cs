namespace ANNModification
{
    internal class Program
    {
        static void PrintMatrix(double[,] matrix)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", matrix[i, j]));
                }
                Console.Write(Environment.NewLine);
            }
        }

        static void PrintMatrix(int[,] matrix)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", matrix[i, j]));
                }
                Console.Write(Environment.NewLine);
            }
        }

        static void Main(string[] args)
        {
            var curNeuralNetwork = new NeuralNetWork(1, 4);

            Console.WriteLine("Synaptic weights before training:");
            PrintMatrix(curNeuralNetwork.SynapsesMatrix);

            var trainingInputs = new double[,] { 
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 }, 
                { 0, 0, 1, 0 }, 
                { 0, 0, 1, 1 }, 
                { 0, 1, 0, 0 }, 
                { 0, 1, 0, 1 }, 
                { 0, 1, 1, 0 }, 
                { 0, 1, 1, 1 }, 
                { 1, 0, 0, 0 }, 
                { 1, 0, 0, 1 }, 
                { 1, 0, 1, 0 }, 
                { 1, 0, 1, 1 }, 
                { 1, 1, 0, 0 }, 
                { 1, 1, 0, 1 }, 
                { 1, 1, 1, 0 }, 
                { 1, 1, 1, 1 },
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 1, 0, 1 },
                { 0, 1, 1, 0 },
                { 0, 1, 1, 1 },
                { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 0 },
                { 1, 0, 1, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 0, 1 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 },
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 }};
            var trainingOutputs = NeuralNetWork.MatrixTranspose(new double[,] { { 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0 } });

            curNeuralNetwork.Train(trainingInputs, trainingOutputs, 1000000);

            Console.WriteLine("\nSynaptic weights after training:");
            PrintMatrix(curNeuralNetwork.SynapsesMatrix);


            // testing neural networks against a new problem 
            var output = curNeuralNetwork.Think(new double[,] { { 1, 0, 0, 0 } });
            var intOutput = curNeuralNetwork.ThresholdOutput(output);
            Console.WriteLine("\nConsidering new problem [1, 0, 0, 0] => :");
            PrintMatrix(intOutput);

            while (true)
            {
                Console.WriteLine("Enter four digits, 0 or 1");
                var input = Console.ReadLine() ?? "";
                var tempList = new List<double>();
                foreach (var ch in input)
                {
                    // learned about this char insanity from chatgpt
                    if (char.IsDigit(ch) && ch - '0' <= 1)
                    {
                        tempList.Add(ch - '0');
                    }
                    else
                    {
                        break;
                    }
                }

                if (tempList.Count != 4) break;
                
                var newInput = tempList.ToArray();

                var tempOutput = curNeuralNetwork.Think(new double[,] { { newInput[0], newInput[1], newInput[2], newInput[3] } });
                var tempIntOutput = curNeuralNetwork.ThresholdOutput(tempOutput);
                Console.WriteLine($"\nConsidering new problem [{newInput[0]}, {newInput[1]}, {newInput[2]}, {newInput[3]}] => :");
                PrintMatrix(tempIntOutput);
            }

        }
    }
}