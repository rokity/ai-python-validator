import { useEffect, useState } from "react";
import type { Route } from "./+types/home";

export function meta({ }: Route.MetaArgs) {
  return [
    { title: "Exercises" },
    { name: "description", content: "Exercises" },
  ];
}

export default function Exercises() {
  const [exercises, setExercises] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/exercises_list', { method: 'GET', mode: 'cors' })
      .then((res) => res.json())
      .then((data) => {
        const listItems = data.map(exercise =>
          <li>
            <p>Title: {exercise['title']} </p>
            <a href={`/exercise/${encodeURIComponent(exercise['id'])}`}>
              <button type="button" className="text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700
               hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 
               font-medium rounded-base text-sm px-4 py-2.5 text-center leading-5">Play</button>
            </a>

            {/* Practice Problem: {exercise['practice problem']}
              Exercise Purpose: {exercise['exercise purpose']}
              Given Input: {exercise['given input']}
              Expected Output: {exercise['expected output']} */}
          </li>

        );
        setExercises(listItems)
      });
  }, []);

  if (!exercises) return <p> Loading</p>;
  return (
    <div>
      <div className="mx-2 flex items-baseline outline-black/5 dark:bg-slate-800 dark:shadow-none dark:-outline-offset-1 dark:outline-white/10">
        <section>
          <div className="text-xl font-medium text-black dark:text-white">Exercises</div>
          <div className="mx-2 pt-8 pl-4 flex items-baseline">
            <ul className="list-decimal">
              {exercises}
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}
