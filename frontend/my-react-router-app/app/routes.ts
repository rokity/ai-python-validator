import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [index("routes/home.tsx"),
    route("exercises", "routes/exercises.tsx"),
    route("exercise/:id", "routes/exercise.tsx")
] satisfies RouteConfig;

