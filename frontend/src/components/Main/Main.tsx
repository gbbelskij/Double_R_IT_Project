import classNames from "classnames";

import { MainProps } from "./Main.props";

import classes from "./Main.module.css";

const Main: React.FC<MainProps> = ({
  children,
  disableHeaderOffset = false,
}) => {
  return (
    <main
      className={classNames(
        classes.Main,
        !disableHeaderOffset && classes.MainWithHeaderOffset
      )}
    >
      {children}
    </main>
  );
};

export default Main;
