#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <SFML/Graphics.hpp>
using namespace std;
using namespace sf;

class Neat{
private:
    vector<float> inputs;
    vector<float> outputs;
    vector<float> hiddens;
    vector<float> inBias;
    vector<float> outBias;
    vector<vector<float>> inWeights;
    vector<vector<float>> outWeights;

    int numIns;
    int numOuts;
    int numHids;
    float fitness;

    static vector<Neat> * pNeats;
    static int generation;
    static int popSize;

    static int sNumIns;
    static int sNumOuts;
    static int sNumHids;
    static Vector2f sWeightsRange;
    static Vector2f sBiasRange;
    static Vector2f sMutationRate;

    static vector<vector<float>> randMatrix(int rows, int cols, Vector2f range);
    static vector<vector<float>> multMatrix(vector<vector<float>> matrixA, vector<vector<float>> matrixB);
    static vector<vector<float>> multMatrix(vector<vector<float>> matrixA, vector<float> matrixB);
    static vector<vector<float>> sumMatrix(vector<vector<float>> matrixA, vector<vector<float>> matrixB);
    static vector<vector<float>> vectorToMatrix(vector<float> vec);
    static vector<float> matrixToVector(vector<vector<float>> matrix);
    static float sigmoid(float value);
public:
    static void defPopulation(vector<Neat> * pNeats, int popSize, int numIns, int numHids, int numOuts, Vector2f weightsRange, Vector2f biasRange, Vector2f mutationRate);
    static void newGeneration();

    Neat(int numIns, int numHids, int numOuts, Vector2f weightsRange, Vector2f biasRange);
    void appendFitness(float value);
    float getFitness();
    void setInputs(vector<float> inputs);
    vector<float> getOutput();
    vector<int> getIntOutput();
};