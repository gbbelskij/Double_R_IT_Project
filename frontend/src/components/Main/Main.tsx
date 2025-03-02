import { MainProps } from "./Main.props";
import classes from "./Main.module.css";

const Main: React.FC<MainProps> = ({ children }) => {
  return <main className={classes.Main}>{children}</main>;
};

export default Main;
