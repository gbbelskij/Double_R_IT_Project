import { useEffect } from "react";
import { useNavigate, useRouteError } from "react-router-dom";

import { useWindowSize } from "@hooks/useWindowSize";

import Main from "@components/Main/Main";
import Logo from "@components/Logo/Logo";

import "./ErrorPage.css";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

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

const ErrorPage: React.FC = () => {
  const error = useRouteError() as RouteError;
  const navigate = useNavigate();
  const { isMobile } = useWindowSize();

  useEffect(() => {
    setTimeout(() => {
      if (error.status === 404) {
        navigate("/", { replace: true });
      }
    }, 1000);
  }, [error, navigate]);

  return (
    <Main disableHeaderOffset>
      <div className="error-container">
        <Logo size={isMobile ? 50 : undefined} />

        <h1 className="error-page--heading">Упс! Что-то пошло не так!</h1>

        <p className="error-page--text">
          Перенаправление на главную страницу...
        </p>

        <BackgroundElements />
      </div>
    </Main>
  );
};

export default ErrorPage;
