import { useEffect } from "react";
import { useNavigate, useRouteError } from "react-router-dom";

interface RouteError {
  data: string;
  error: {
    columnNumber: number;
    fileName: string;
    lineNumber: number;
    message: string;
    stack: string;
  };
  internal: boolean;
  status: number;
  statusText: string;
}

export default function ErrorPage() {
  const error = useRouteError() as RouteError;
  const navigate = useNavigate();

  useEffect(() => {
    setTimeout(() => {
      if (error.status === 404) {
        navigate("/", { replace: true });
      }
    }, 2000);
  }, [error, navigate]);

  return (
    <div>
      <h1>Упс! Что-то пошло не так.</h1>
      <p>Перенаправление на главную страницу...</p>
    </div>
  );
}
