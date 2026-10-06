import { useLocation, useSearchParams } from "react-router";
import type { Route } from "./+types/home";
import { useCallback, useEffect, useState } from "react";
import ReactCodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";

export function meta({ }: Route.MetaArgs) {
    return [
        { title: "Exercise" },
        { name: "description", content: "Exercise" },
    ];
}


export default function Exercise() {
    let exercise_id = useLocation().pathname.split("/")[2];
    const [exercise, setExercise] = useState(null);
    const [value, setValue] = useState("# Write here your solution");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<string | null>(null);

    const onChange = useCallback((val, viewUpdate) => {
        setValue(val);
    }, []);

    useEffect(() => {
        fetch(`http://localhost:8000/get_exercise?id=${exercise_id}`, { method: 'GET', mode: 'cors' })
            .then((res) => res.json())
            .then((data) => {
                setExercise(data)
            });
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResult(null);
        try {
            const res = await fetch(`http://localhost:8000/validate_exercise/`, {
                method: 'POST',
                mode: 'cors',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: exercise_id, code: value }),
            });
            const data = await res.json();
            if (data["syntax"]=="INVALID")
                setResult("SYNTAX INVALID");
            else if (data["correctness"]=="INCORRECT")
                setResult("EXERCISE INVALID");
            else
                setResult("EXERCISE VALID");
        } catch (err) {
            console.error(err);
            setResult("ERROR");
        } finally {
            setLoading(false);
        }
    };
    if (!exercise) return <p> Loading</p>;
    return (
        <div>
            <form onSubmit={handleSubmit} >
                <div className="mx-2 flex items-baseline outline-black/5 dark:bg-slate-800 dark:shadow-none dark:-outline-offset-1 dark:outline-white/10">
                    <section>
                        <div className="text-xl font-medium text-black dark:text-white">Title: {exercise['title']}</div>
                        <div className="mx-2 pt-8 pl-4 flex items-baseline">
                            <ul className="list-disc">
                                <li>Practice Problem: {exercise['practice problem']}</li>
                                <li>Exercise Purpose: {exercise['exercise purpose']}</li>
                                <li>Given Input: {exercise['given input']}</li>
                                <li>Expected Output: {exercise['expected output']}</li>
                            </ul>
                        </div>
                    </section>
                </div>
                <div className="pt-4">
                    <div className="text-xl font-medium text-black dark:text-white">Solution</div>
                </div>
                <div className="pl-8 pt-8">
                    <ReactCodeMirror
                        value={value}
                        theme="dark"
                        extensions={[python()]}
                        height="auto"
                        width="800px"
                        margin-left="10px"
                        onChange={onChange}
                    />
                </div>
                <div className="pl-8 pt-8">
                    <button type="submit" className="text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700
               hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 
               font-medium rounded-base text-sm px-4 py-2.5 text-center leading-5 disabled:opacity-50" disabled={loading}>Submit</button>
                    {result && (
                        <span className={`ml-4 px-3 py-1 rounded font-semibold text-white ${result.includes(" VALID") ? "bg-green-600" : result.includes(" INVALID") ? "bg-red-600" : "bg-gray-600"}`}>
                            {result === "ERROR" ? "Error contacting the server" : result}
                        </span>
                    )}
                </div>
            </form>
            {loading && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                    <div className="flex flex-col items-center gap-4 rounded-lg bg-white p-8 shadow-lg dark:bg-slate-800">
                        <div className="h-10 w-10 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
                        <p className="text-black dark:text-white">Validating your solution...</p>
                    </div>
                </div>
            )}
        </div>
    );
}