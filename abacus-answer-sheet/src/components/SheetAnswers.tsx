/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState } from 'react';
import type { ChangeEvent } from 'react';
import { Upload, Calculator, CheckCircle, XCircle } from 'lucide-react';

type Problem = {
  numbers: number[];
  correct_answer: number;
};

type APIResponse = {
  results: Problem[];
  total_questions: number;
  correct_answers: string;
};

const MathWorksheetSolver: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<APIResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResponse(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(
        "https://abacus-sheet-solver.onrender.com/check-answers",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to solve worksheet");
      }

      const data: APIResponse = await res.json();
      setResponse(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderWorksheet = () => {
    if (!response?.results) return null;

    const problems = response.results;
    const rows: Problem[][] = [];
    const problemsPerRow = 20;

    for (let i = 0; i < problems.length; i += problemsPerRow) {
      rows.push(problems.slice(i, i + problemsPerRow));
    }

    return (
      <div className="mt-8 w-full">
        <div className="bg-white rounded-lg shadow-lg p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6 pb-4 border-b-2 border-gray-300">
            <div className="flex items-center gap-3">
              <img
                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='45' fill='%232563eb'/%3E%3Ctext x='50' y='70' font-size='60' text-anchor='middle' fill='white' font-family='Arial' font-weight='bold'%3EA%3C/text%3E%3C/svg%3E"
                alt="Logo"
                className="w-12 h-12"
              />
              <div>
                <h2 className="text-2xl font-bold text-gray-800">Trisha ABACUS</h2>
                <p className="text-sm text-gray-600">Counting Made Magical</p>
              </div>
            </div>
            <div className="text-right">
              <h3 className="text-xl font-bold text-gray-800">ABC Category</h3>
              <p className="text-sm text-gray-600">Solved Worksheet</p>
            </div>
          </div>

          {/* Rows */}
          {rows.map((row, rowIndex) => (
            <div key={rowIndex} className="mb-8">
              <div className="flex items-center gap-2 mb-3">
                <div className="bg-blue-600 text-white px-3 py-1 rounded font-semibold">
                  Row {rowIndex + 1}
                </div>
                <div className="text-sm text-gray-600">
                  Problems {rowIndex * problemsPerRow + 1} -{" "}
                  {Math.min((rowIndex + 1) * problemsPerRow, problems.length)}
                </div>
              </div>

              <div className="grid grid-cols-10 gap-3">
                {row.map((problem, colIndex) => (
                  <ProblemCard
                    key={colIndex}
                    problem={problem}
                    problemNumber={rowIndex * problemsPerRow + colIndex + 1}
                  />
                ))}
              </div>
            </div>
          ))}

          {/* Footer */}
          <div className="mt-6 pt-4 border-t-2 border-gray-300 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="text-gray-700 font-medium">
                Total Problems Solved: {response.total_questions}
              </span>
            </div>
            <div className="text-sm text-gray-500">
              {response.correct_answers}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2 flex items-center justify-center gap-3">
            <Calculator className="w-10 h-10 text-blue-600" />
            Math Worksheet Solver
          </h1>
          <p className="text-gray-600">
            Upload your worksheet and get instant solutions
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
          <div className="flex flex-col items-center gap-6">
            <div className="w-full max-w-md">
              <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-12 h-12 text-gray-400 mb-3" />
                  <p className="mb-2 text-sm text-gray-500">
                    <span className="font-semibold">Click to upload</span> or drag
                    and drop
                  </p>
                  <p className="text-xs text-gray-500">PNG, JPG or JPEG</p>
                  {file && (
                    <p className="mt-2 text-sm text-blue-600 font-medium">
                      Selected: {file.name}
                    </p>
                  )}
                </div>
                <input
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </label>
            </div>

            {previewUrl && (
              <div className="w-full max-w-md">
                <p className="text-sm text-gray-600 mb-2 font-medium">Preview:</p>
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full rounded-lg border-2 border-gray-200"
                />
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={!file || loading}
              className="px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-md hover:shadow-lg"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Calculator className="w-5 h-5" />
                  Solve Worksheet
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-red-800">Error</p>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          )}
        </div>

        {renderWorksheet()}
      </div>
    </div>
  );
};

export default MathWorksheetSolver;

type ProblemCardProps = {
  problem: Problem;
  problemNumber: number;
};

const ProblemCard: React.FC<ProblemCardProps> = ({ problem, problemNumber }) => {
  return (
    <div className="bg-gradient-to-br from-white to-blue-50 rounded-lg shadow-md p-3 border-2 border-blue-200 hover:shadow-lg transition-all">
      <div className="text-xs text-gray-500 font-medium mb-2 text-center">
        #{problemNumber}
      </div>

      <div className="space-y-1 mb-3">
        {problem.numbers.map((num: number, idx: number) => (
          <div
            key={idx}
            className="bg-white rounded px-2 py-1 text-center border border-gray-200"
          >
            <span
              className={`font-mono font-semibold text-sm ${
                num < 0 ? "text-red-600" : "text-gray-800"
              }`}
            >
              {num}
            </span>
          </div>
        ))}
      </div>

      <div className="border-t-2 border-blue-400 mb-2"></div>

      <div className="bg-blue-600 text-white rounded px-2 py-2 text-center shadow-md">
        <span className="font-mono font-bold text-base">
          {problem.correct_answer}
        </span>
      </div>
    </div>
  );
};
