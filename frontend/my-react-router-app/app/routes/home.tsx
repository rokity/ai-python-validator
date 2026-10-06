import type { Route } from "./+types/home";

export function meta({ }: Route.MetaArgs) {
  return [
    { title: "Home Page" },
    { name: "description", content: "Home Page" },
  ];
}

export default function Home() {
  return (
    <div>
      <div className="mx-2 flex items-baseline outline-black/5 dark:bg-slate-800 dark:shadow-none dark:-outline-offset-1 dark:outline-white/10">
        <section>
          <div className="text-xl font-medium text-black dark:text-white">E-Learning Platform based on LLM coding models</div>
          <p> Here you can find different courses and exercise to learn Python and became a Python developer.</p>
        </section>
      </div>
      <div className="mx-2 pt-8 pl-4 flex items-baseline">
        <ul className="list-decimal">
          <li><a className="no-underline md:underline ..." href="/exercises"> Python Basic Exercise for Beginners: 40 Coding Problems </a></li>
          <li><a className="no-underline md:underline ..." href="/exercises"> Python Advanced Exercise : 30 Coding Problems </a></li>
        </ul>
      </div>
    </div>
  );
}