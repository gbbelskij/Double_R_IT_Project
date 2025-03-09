import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { Error, Home, Login, Profile, Registration } from "@pages";

const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <Home />,
      errorElement: <Error />,
    },
    {
      path: "/login",
      element: <Login />,
    },
    {
      path: "/profile",
      element: <Profile />,
    },
    {
      path: "/registration",
      element: <Registration />,
    },
  ],
  { basename: import.meta.env.BASE_URL }
);

export default function Router() {
  return <RouterProvider router={router} />;
}
