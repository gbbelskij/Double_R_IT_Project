import { useEffect, useRef } from "react";
import { useNavigate, useRouteError } from "react-router-dom";

import { useWindowSize } from "@hooks/useWindowSize";

import Main from "@components/Main/Main";
import Logo from "@components/Logo/Logo";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import "./ErrorPage.css";

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

  const { isMobile, isSmallMobile } = useWindowSize();

  const sectionRef = useRef(null);

  useEffect(() => {
    setTimeout(() => {
      if (error.status === 404) {
        navigate("/", { replace: true });
      }
    }, 1000);
  }, [error, navigate]);

  return (
    <Main disableHeaderOffset>
      <div className="error-container" ref={sectionRef}>
        <Logo size={isMobile ? 50 : undefined} />

        <h1 className="error-page--heading">Упс! Что-то пошло не так!</h1>

        <p className="error-page--text">
          Перенаправление на главную страницу...
        </p>
      </div>

      <BackgroundElements
        targetRef={sectionRef}
        blobsSize={isMobile ? (isSmallMobile ? 150 : 200) : 300}
      />
    </Main>
  );
};

export default ErrorPage;
