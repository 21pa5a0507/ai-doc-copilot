import React from "react";

export default function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) {
    return (
      <div className="w-full p-4 text-gray-500 text-center">
        No sources available
      </div>
    );
  }

  return (
    <div className="w-full h-full overflow-y-auto p-4 bg-gray-50 border-l">
      <h2 className="text-lg font-semibold mb-4">Sources</h2>

      <div className="flex flex-col gap-4">
        {sources.map((src, index) => (
          <div
            key={index}
            className="p-4 bg-white rounded-2xl shadow-sm border hover:shadow-md transition"
          >
            {/* Title */}
            <a
              href={src.url}
              target="_blank"
              rel="noreferrer"
              className="text-blue-600 font-medium hover:underline"
            >
              {index + 1}. {src.title || "Untitled Source"}
            </a>

            {/* URL */}
            <p className="text-xs text-gray-400 break-all mt-1">
              {src.url}
            </p>

            {/* Content Preview */}
            <p className="text-sm text-gray-700 mt-2">
              {src.content
                ? src.content.slice(0, 180) + "..."
                : "No preview available"}
            </p>

            {/* Optional Score */}
            {src.score && (
              <p className="text-xs text-green-600 mt-2">
                Relevance Score: {src.score.toFixed(2)}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}