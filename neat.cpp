#include "./neat.hpp"

int Neat::popSize;
int Neat::sNumIns;
int Neat::sNumOuts;
int Neat::sNumHids;
int Neat::generation;
Vector2f Neat::sBiasRange;
Vector2f Neat::sWeightsRange;
Vector2f Neat::sMutationRate;
vector<Neat> * Neat::pNeats;

void Neat::defPopulation(vector<Neat> * pNeats, int popSize, int numIns, int numHids, int numOuts, Vector2f weightsRange, Vector2f biasRange,  Vector2f mutationRate){
    pNeats = pNeats;
    popSize = popSize;
    sNumIns = numIns;
    sNumOuts = numOuts;
    sNumHids = numHids;
    sWeightsRange = weightsRange;
    sBiasRange = biasRange;
    sMutationRate = mutationRate;

    for (int i = 0; i < popSize; i++){
        pNeats->push_back(Neat(sNumIns, sNumHids, sNumOuts, sWeightsRange, sBiasRange));
    }

    generation = 1;
}

void Neat::newGeneration()
{
    int indexBetter = 0;
    
    if (pNeats->size() != 1) {
        for (int i = 1; i < pNeats->size(); i++){
            if (pNeats->at(i).getFitness() > pNeats->at(indexBetter).getFitness()){
                indexBetter = i;
            }
        }

        if (indexBetter != 0) {
            pNeats->insert(pNeats->begin() + indexBetter, pNeats->at(indexBetter));
        }

        for (int i = pNeats->size() - 1; i > 0; i--){
            pNeats->pop_back();
        }  
    }

    for (int i = 1; i < popSize; i++){
        pNeats->push_back(Neat(sNumIns, sNumHids, sNumOuts, sWeightsRange, sBiasRange));

        pNeats->at(i).inWeights = sumMatrix(
            pNeats->at(0).inWeights,
            randMatrix(pNeats->at(0).inWeights.size(), pNeats->at(0).inWeights.at(0).size(), sMutationRate)
        );
    }
}
Neat::Neat(int numIns, int numHids, int numOuts, Vector2f weightsRange, Vector2f biasRange)
{
    srand((unsigned)time(NULL));

    this->inputs = inputs;
    this->numHids = numHids;
    this->numOuts = numOuts;
    this->numIns = numIns;

    this->inWeights = randMatrix(this->numHids, this->numIns, weightsRange);
    this->outWeights = randMatrix(this->numOuts, this->numHids, weightsRange);

    this->inBias = matrixToVector(randMatrix(1, this->numHids, biasRange));
    this->outBias = matrixToVector(randMatrix(1, this->numOuts, biasRange));
}

vector<vector<float>> Neat::multMatrix(vector<vector<float>> matrixA, vector<vector<float>> matrixB)
{
    vector<vector<float>> matrixC;
    float value;

    if (matrixA.at(0).size() == matrixB.size())
    {
        for (int i = 0; i < matrixA.size(); i++)
        {
            matrixC.push_back(vector<float>());
            for (int j = 0; j < matrixB.at(0).size(); j++)
            {
                value = 0;
                for (int k = 0; k < matrixB.size(); k++)
                {
                    value += matrixA.at(i).at(k) * matrixB.at(k).at(j);
                }

                value = round(value * 100) / 100;

                matrixC.at(i).push_back(value);
            }
        }
    }
    else
    {
        cerr << "ERROR!! YOU CAN'T MULTIPLY A " << matrixA.size() << "x" << matrixA.at(0).size() << " MATRIX BY A " << matrixB.size() << "x" << matrixB.at(0).size() << endl;
    }

    return matrixC;
}

vector<vector<float>> Neat::multMatrix(vector<vector<float>> matrixA, vector<float> matrixB)
{
    return multMatrix(matrixA, vectorToMatrix(matrixB));
}

vector<vector<float>> Neat::sumMatrix(vector<vector<float>> matrixA, vector<vector<float>> matrixB)
{
    vector<vector<float>> matrixC;

    if (matrixA.size() == matrixB.size() && matrixA.at(0).size() == matrixB.at(0).size())
    {
        for (int i = 0; i < matrixA.size(); i++)
        {
            matrixC.push_back(vector<float>());
            for (int j = 0; j < matrixA.at(i).size(); j++)
            {
                matrixC.at(i).push_back(matrixA.at(i).at(j) + matrixA.at(i).at(j));
            }
        }
    }
    else
    {
        cerr << "ERROR!! YOU CAN'T SUM A " << matrixA.size() << "x" << matrixA.at(0).size() << " MATRIX AND A " << matrixB.size() << "x" << matrixB.at(0).size() << endl;
    }
    return matrixC;
}

vector<vector<float>> Neat::randMatrix(int rows, int cols, Vector2f range)
{
    vector<vector<float>> matrix;

    for (int i = 0; i < rows; i++)
    {
        matrix.push_back(vector<float>());
        for (int j = 0; j < cols; j++)
        {
            // Getting a random float in range
            matrix.at(i).push_back(round(
                                       (float(rand()) /
                                            float(RAND_MAX / (range.x + (-range.y))) -
                                        (-range.y)) *
                                       100) /
                                   100);
        }
    }

    return matrix;
}

vector<vector<float>> Neat::vectorToMatrix(vector<float> vec)
{
    vector<vector<float>> matrix;

    for (int i = 0; i < vec.size(); i++)
    {
        matrix.push_back(vector<float>());
        matrix.at(i).push_back(vec.at(i));
    }

    return matrix;
}

vector<float> Neat::matrixToVector(vector<vector<float>> matrix)
{
    vector<float> vec;

    for (auto &i : matrix)
    {
        for (auto &j : i)
        {
            vec.push_back(j);
        }
    }

    return vec;
}

float Neat::sigmoid(float value)
{
    return float(round(1 / (1 + exp(-value)) * 100) / 100);
}

void Neat::setInputs(vector<float> inputs)
{
    this->inputs.clear();
    for (auto &i : inputs){
        this->inputs.push_back(i);
    }
}

void Neat::appendFitness(float value)
{
    this->fitness += value;
}

float Neat::getFitness()
{
    return this->fitness;
}

vector<float> Neat::getOutput()
{
    this->hiddens = matrixToVector(sumMatrix(
        multMatrix(this->inWeights, this->inputs),
        vectorToMatrix(this->inBias)));
    this->outputs = matrixToVector(sumMatrix(
        multMatrix(this->outWeights, this->hiddens),
        vectorToMatrix(this->outBias)));

    for (auto &i : this->outputs)
    {
        i = sigmoid(i);
    }

    return this->outputs;
}

vector<int> Neat::getIntOutput()
{
    vector<int> intOutput;
    this->getOutput();

    for (auto &i : this->outputs)
    {
        intOutput.push_back(int(round(i)));
    }

    return intOutput;
}
